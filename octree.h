#pragma once

#include "types.h"
#include "raytracer.h"
#include "mesh.h"

namespace Tmpl8 {

    struct OctreeNode {
		float3 aabbMin, aabbMax;
		uint firstChildIdx;
		uint triCount;

        bool isLeaf() const { return triCount > 0; }
    };

    __declspec(align(64)) class Octree : public ISpatialStructure {
        public:
            Octree() = default;
            void Build( Mesh* mesh ) override;
            void Intersect( Ray& ray, uint instanceIdx ) override;

			// interface implementation
			const void* GetNodes() const override { return nodes; }
			const uint* GetTriIndices() const override { return triIdx; }
			size_t GetNodeCount() const override { return nodesUsed; }
			size_t GetTriIndexCount() const override { return triCount; }
			aabb GetRootBounds() const override;
        
        private:
            void Subdivide( uint nodeIdx, uint depth, const float3& pMin, const float3& pMax );
            void UpdateNodeBounds( uint nodeIdx );

            class Mesh* mesh = 0;
            uint* triIdx = 0;
			uint triCount = 0;
            OctreeNode* nodes = 0;
            uint nodesUsed = 0;
            uint maxDepth = 8;
    };
}