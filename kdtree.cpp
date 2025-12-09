#include "precomp.h"
#include "kdtree.h"

const int MAX_KDTREE_NODES = 50000000;


void Kdtree::Build(Mesh* triMesh)
{
    mesh = triMesh;
    triCount = mesh->triCount;
    nodes = (KdtreeNode*)_aligned_malloc(sizeof(KdtreeNode) * MAX_KDTREE_NODES, 64);
    triIdx = new uint[mesh->triCount];

    for (int i = 0; i < mesh->triCount; i++)
    {
        triIdx[i] = i;
        // 顺便计算质心，后面 split 要用
        mesh->tri[i].centroid = (mesh->tri[i].vertex0 +
            mesh->tri[i].vertex1 +
            mesh->tri[i].vertex2) * (1.0f / 3.0f);
    }

    memset(nodes, 0, sizeof(KdtreeNode) * MAX_KDTREE_NODES);
    nodesUsed = 1;

    // 根节点：叶子，包含所有三角形
    KdtreeNode& root = nodes[0];
    root.firstChildIdx = 0;              // triIdx 起始位置
    root.triCount = mesh->triCount; // 当前节点三角形数

    // 先算一次根节点包围盒
    UpdateNodeBounds(0);

    // 用包围盒做第一次划分
    Subdivide(0, 0);
}


void Kdtree::UpdateNodeBounds(uint nodeIdx)
{
    KdtreeNode& node = nodes[nodeIdx];
    node.aabbMin = float3(1e30f);
    node.aabbMax = float3(-1e30f);

    for (uint i = 0; i < node.triCount; i++)
    {
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


// 新版：真正的 K-d tree 划分
void Kdtree::Subdivide(uint nodeIdx, uint depth)
{
    KdtreeNode& node = nodes[nodeIdx];

    // 叶子终止条件
    if (node.triCount <= 2 || depth >= maxDepth)
    {   UpdateNodeBounds(nodeIdx);
        return;
    }
        

    // 1. 先根据当前三角形更新一下包围盒（保证 aabbMin/Max 正确）
    UpdateNodeBounds(nodeIdx);

    // 2. 选择划分轴：取包围盒最长的一轴
    float3 extent = node.aabbMax - node.aabbMin;
    uint axis = 0;
    if (extent.y > extent.x && extent.y > extent.z) axis = 1;
    else if (extent.z > extent.x && extent.z > extent.y) axis = 2;

    // 3. 选择划分位置：这里用包围盒中点（也可以用质心中位数，效果更好）
    float splitPos = node.aabbMin[axis] + 0.5f * extent[axis];

    // 4. 按 axis + splitPos 把三角形分到左右两个区间
    uint triStart = node.firstChildIdx;
    uint triEnd = triStart + node.triCount;

    uint leftCount = 0;
    uint rightCount = 0;

    uint triNum = node.triCount;
    uint* temp = new uint[triNum];

    // 先把 <= split 的放前面，> split 的放后面
    for (uint i = triStart; i < triEnd; i++)
    {
        uint tIdx = triIdx[i];
        float c = mesh->tri[tIdx].centroid[axis];

        if (c <= splitPos)
        {
            temp[leftCount++] = tIdx;
        }
        else
        {
            temp[triNum - 1 - rightCount] = tIdx;
            rightCount++;
        }
    }

    // 如果所有三角形都在同一侧，说明这次划分无效，就直接当叶子使用
    if ((leftCount == 0 && rightCount == node.triCount) ||(rightCount == 0 && leftCount == node.triCount))
    {
        delete[] temp;
        UpdateNodeBounds(nodeIdx);
        return;
    }

    // 把重新排序后的索引写回 triIdx[triStart ... triEnd)
    memcpy(&triIdx[triStart], temp, triNum * sizeof(uint));
    delete[] temp;

    // 5. 检查节点数量是否溢出
    if (nodesUsed + 2 > MAX_KDTREE_NODES)
        return;

    uint leftChildIdx = nodesUsed++;
    uint rightChildIdx = nodesUsed++;

    // 6. 把当前节点改成内部节点
    node.firstChildIdx = leftChildIdx; // 指向左孩子
    node.triCount = 0;            // triCount==0 表示内部节点
    node.axis = axis;
    node.splitPos = splitPos;

    // 7. 初始化左右子节点（它们现在是叶子，里面有各自的三角形区间）
    KdtreeNode& left = nodes[leftChildIdx];
    KdtreeNode& right = nodes[rightChildIdx];

    left.firstChildIdx = triStart;         // 左边三角形起始索引
    left.triCount = leftCount;        // 左边三角形数
    right.firstChildIdx = triStart + leftCount;
    right.triCount = rightCount;

    // 计算左右节点自己的包围盒
    UpdateNodeBounds(leftChildIdx);
    UpdateNodeBounds(rightChildIdx);

    // 8. 递归划分子节点
    Subdivide(leftChildIdx, depth + 1);
    Subdivide(rightChildIdx, depth + 1);
}


aabb Kdtree::GetRootBounds() const
{
    aabb bounds;
    bounds.bmin = nodes[0].aabbMin;
    bounds.bmax = nodes[0].aabbMax;
    return bounds;
}

void Kdtree::Intersect(Ray& ray, uint instanceIdx) {
    KdtreeNode* node = &nodes[0];
    KdtreeNode* stack[64];
    uint stackPtr = 0;

    while (true) {
        if (node->isLeaf()) {
            if (stackPtr == 0) break;
            else {
                node = stack[--stackPtr];
            }
            continue;
        }

        uint firstChild = node->firstChildIdx;
        for (int i = 7; i >= 0; i--) {
            KdtreeNode& child = nodes[firstChild + i];
        }

        if (stackPtr == 0) break;
        else {
            node = stack[--stackPtr];
        }
    }
}