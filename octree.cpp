#include "precomp.h"
#include "octree.h"

// Note: We do not need "raytracer.h" or "types.h" as they are covered by precomp.h and bvh.h

const int MAX_OCTREE_NODES = 50000000;
void Octree::Build(Mesh* triMesh)
{
    mesh = triMesh;
    triCount = mesh->triCount;
    nodes = (OctreeNode*)_aligned_malloc(sizeof(OctreeNode) * MAX_OCTREE_NODES, 64);
    triIdx = new uint[mesh->triCount];

    for (int i = 0; i < mesh->triCount; i++)
        triIdx[i] = i;

    nodesUsed = 1;
    memset(nodes, 0, sizeof(OctreeNode) * MAX_OCTREE_NODES);

    OctreeNode& root = nodes[0];
    root.firstChildIdx = 0;
    root.triCount = mesh->triCount;

    // Calculate mesh bounds
    aabb meshBounds;
    for (int i = 0; i < mesh->triCount; i++) {
        mesh->tri[i].centroid = (mesh->tri[i].vertex0 + mesh->tri[i].vertex1 + mesh->tri[i].vertex2) * (1.0f / 3.0f);
        meshBounds.grow(mesh->tri[i].vertex0);
        meshBounds.grow(mesh->tri[i].vertex1);
        meshBounds.grow(mesh->tri[i].vertex2);
    }

    Subdivide(0, 0, meshBounds.bmin, meshBounds.bmax);
}

void Octree::UpdateNodeBounds(uint nodeIdx) {
    OctreeNode& node = nodes[nodeIdx];
    node.aabbMin = float3(1e30f);
    node.aabbMax = float3(-1e30f);

    for (uint i = 0; i < node.triCount; i++) {
        uint leafTriIdx = triIdx[node.firstChildIdx + i];
        Tri& leafTri = mesh->tri[leafTriIdx];
        node.aabbMin = fminf(node.aabbMin, leafTri.vertex0);
        node.aabbMin = fminf(node.aabbMin, leafTri.vertex1);
        node.aabbMin = fminf(node.aabbMin, leafTri.vertex2);
        node.aabbMax = fmaxf(node.aabbMax, leafTri.vertex0);
        node.aabbMax = fmaxf(node.aabbMax, leafTri.vertex1);
        node.aabbMax = fmaxf(node.aabbMax, leafTri.vertex2);
    }
}

void Octree::Subdivide(uint nodeIdx, uint depth, const float3& pMin, const float3& pMax) {
    OctreeNode& node = nodes[nodeIdx];

    if (node.triCount <= 2 || depth >= maxDepth) {
        UpdateNodeBounds(nodeIdx);
        return;
    }

    float3 center = (pMin + pMax) * 0.5f;

    int counts[8] = { 0 };
    for (uint i = 0; i < node.triCount; i++)
    {
        uint tIdx = triIdx[node.firstChildIdx + i];
        float3 c = mesh->tri[tIdx].centroid;
        int octant = (c.x > center.x) | ((c.y > center.y) << 1) | ((c.z > center.z) << 2);
        counts[octant]++;
    }

    int offsets[8];
    offsets[0] = 0;
    for (int i = 1; i < 8; ++i) offsets[i] = offsets[i - 1] + counts[i - 1];

    uint* temp_indices = new uint[node.triCount];
    int C[8] = { 0 };
    for (uint i = 0; i < node.triCount; ++i)
    {
        uint tIdx = triIdx[node.firstChildIdx + i];
        float3 c = mesh->tri[tIdx].centroid;
        int octant = (c.x > center.x) | ((c.y > center.y) << 1) | ((c.z > center.z) << 2);
        temp_indices[offsets[octant] + C[octant]] = tIdx;
        C[octant]++;
    }
    memcpy(&triIdx[node.firstChildIdx], temp_indices, node.triCount * sizeof(uint));
    delete[] temp_indices;

    // Check if we failed to split any triangles (infinite recursion prevention)
    for (int i = 0; i < 8; i++) if (counts[i] == node.triCount)
    {
        UpdateNodeBounds(nodeIdx);
        return;
    }

    if (nodesUsed + 8 > MAX_OCTREE_NODES) return;

    // --- CRITICAL FIX START ---
    // Save the pointer to the triangles BEFORE overwriting firstChildIdx
    uint triangleStartIndex = node.firstChildIdx;
    // -------------------------

    uint firstChild = nodesUsed;
    nodesUsed += 8;

    // Now we can overwrite this to point to the children nodes
    node.firstChildIdx = firstChild;
    node.triCount = 0;

    for (int i = 0; i < 8; i++) {
        OctreeNode& child = nodes[firstChild + i];

        // Use the SAVED triangleStartIndex, not node.firstChildIdx
        child.firstChildIdx = triangleStartIndex + offsets[i];
        child.triCount = counts[i];

        float3 childMin = { (i & 1) ? center.x : pMin.x, (i & 2) ? center.y : pMin.y, (i & 4) ? center.z : pMin.z };
        float3 childMax = { (i & 1) ? pMax.x : center.x, (i & 2) ? pMax.y : center.y, (i & 4) ? pMax.z : center.z };

        if (child.triCount > 0) {
            Subdivide(firstChild + i, depth + 1, childMin, childMax);
        }
        else {
            child.aabbMin = childMin;
            child.aabbMax = childMax;
            child.firstChildIdx = 0;
        }
    }

    // Only fit bounds to populated children
    node.aabbMin = float3(1e30f);
    node.aabbMax = float3(-1e30f);
    for (int i = 0; i < 8; i++) {
        OctreeNode& child = nodes[firstChild + i];
        if (child.triCount > 0 || child.firstChildIdx > 0) {
            node.aabbMin = fminf(node.aabbMin, child.aabbMin);
            node.aabbMax = fmaxf(node.aabbMax, child.aabbMax);
        }
    }
}

aabb Octree::GetRootBounds() const
{
    aabb bounds;
    bounds.bmin = nodes[0].aabbMin;
    bounds.bmax = nodes[0].aabbMax;
    return bounds;
}

void Octree::Intersect(Ray& ray, uint instanceIdx) {
    OctreeNode* node = &nodes[0];
    OctreeNode* stack[64];
    uint stackPtr = 0;

    while (true) {
        if (node->isLeaf()) {
            for (uint i = 0; i < node->triCount; i++) {
                // CPU intersection not needed for GPGPU assignment
            }
            if (stackPtr == 0) break;
            else {
                node = stack[--stackPtr];
            }
            continue;
        }

        uint firstChild = node->firstChildIdx;
        for (int i = 7; i >= 0; i--) {
            OctreeNode& child = nodes[firstChild + i];
            if (child.triCount > 0 || child.firstChildIdx > 0) {
                // CPU intersection not needed for GPGPU assignment
            }
        }

        if (stackPtr == 0) break;
        else {
            node = stack[--stackPtr];
        }
    }
}