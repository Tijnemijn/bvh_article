#pragma once
#include "bvh.h" 

namespace Tmpl8 {

    struct KdtreeNode
    {
        float3 aabbMin;
        float3 aabbMax;

        uint   firstChildIdx; // 叶子: triIdx 起始位置; 内部: 左孩子节点 index
        uint   triCount;      // >0 表示叶子; ==0 表示内部节点

        uint   axis;          // 内部节点使用: 0=x,1=y,2=z
        float  splitPos;      // 内部节点使用: 切分平面位置

        bool isLeaf() const { return triCount > 0; }
    };

    __declspec(align(64)) class Kdtree {
    public:
        Kdtree() = default;

        // Removed 'override' keywords
        void Build(Mesh* mesh);
        void Intersect(Ray& ray, uint instanceIdx);

        // Getters for GPU data transfer
        const KdtreeNode* GetNodes() const { return nodes; }
        const uint* GetTriIndices() const { return triIdx; }
        size_t GetNodeCount() const { return nodesUsed; }
        size_t GetTriIndexCount() const { return triCount; }
        aabb GetRootBounds() const;

    private:
        void Kdtree::Subdivide(uint nodeIdx, uint depth);

        void UpdateNodeBounds(uint nodeIdx);

        class Mesh* mesh = 0;
        uint* triIdx = 0;
        uint triCount = 0;
        KdtreeNode* nodes = 0;
        uint nodesUsed = 0;
        uint maxDepth = 8;
    };
}