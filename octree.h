#pragma once
#include "bvh.h" 

namespace Tmpl8 {

    struct OctreeNode {
        float3 aabbMin, aabbMax;
        uint firstChildIdx;
        uint triCount;

        bool isLeaf() const { return triCount > 0; }
    };

    __declspec(align(64)) class Octree {
    public:
        Octree() = default;

        // Removed 'override' keywords
        void Build(Mesh* mesh);
        void Intersect(Ray& ray, uint instanceIdx);

        // Getters for GPU data transfer
        const OctreeNode* GetNodes() const { return nodes; }
        const uint* GetTriIndices() const { return triIdx; }
        size_t GetNodeCount() const { return nodesUsed; }
        size_t GetTriIndexCount() const { return triCount; }
        aabb GetRootBounds() const;

    private:
        void Subdivide(uint nodeIdx, uint depth, const float3& pMin, const float3& pMax);
        void UpdateNodeBounds(uint nodeIdx);

        class Mesh* mesh = 0;
        uint* triIdx = 0;
        uint triCount = 0;
        OctreeNode* nodes = 0;
        uint nodesUsed = 0;
        uint maxDepth = 8;
    };
}