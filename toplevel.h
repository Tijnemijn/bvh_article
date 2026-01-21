#pragma once

#include <vector>

// enable the use of SSE in the AABB intersection function
#define USE_SSE

// bin count
#define BINS 8

namespace Tmpl8
{

	// minimalist triangle struct
	struct Tri { float3 vertex0, vertex1, vertex2; float3 centroid; };

	// ray struct, prepared for SIMD AABB intersection
	__declspec(align(64)) struct Ray
	{
		Ray() { O4 = D4 = rD4 = _mm_set1_ps(1); }
		union { struct { float3 O; float dummy1; }; __m128 O4; };
		union { struct { float3 D; float dummy2; }; __m128 D4; };
		union { struct { float3 rD; float dummy3; }; __m128 rD4; };
		float t = 1e30f;
	};

	// minimalist AABB struct with grow functionality
	struct aabb
	{
		float3 bmin = 1e30f, bmax = -1e30f;
		void grow(float3 p) { bmin = fminf(bmin, p); bmax = fmaxf(bmax, p); }
		void grow(aabb& b) { if (b.bmin.x != 1e30f) { grow(b.bmin); grow(b.bmax); } }
		float area() const
		{
			float3 e = bmax - bmin; // box extent
			return e.x * e.y + e.y * e.z + e.z * e.x;
		}
	};

	// 32-byte BVH node struct
	struct BVHNode
	{
		union { struct { float3 aabbMin; uint leftFirst; }; __m128 aabbMin4; };
		union { struct { float3 aabbMax; uint triCount; }; __m128 aabbMax4; };
		bool isLeaf() { return triCount > 0; }
		float CalculateNodeCost()
		{
			float3 e = aabbMax - aabbMin; // extent of the node
			return (e.x * e.y + e.y * e.z + e.z * e.x) * triCount;
		}
	};

	// bvh 
	class BVH
	{
	public:
		BVH() = default;
		BVH(char* triFile, int N);
		void Build();
		void Refit();
		void SetTransform(mat4& transform);
		void Intersect(Ray& ray, uint nodeIdx = 0);
		uint GetTriCount() const { return triCount; }

		// Helpers for Re-braiding
		BVHNode* GetNode(uint idx) { return &bvhNode[idx]; }
		mat4& GetTransform() { return transform; }

	private:
		void Subdivide(uint nodeIdx);
		void UpdateNodeBounds(uint nodeIdx);
		float FindBestSplitPlane(BVHNode& node, int& axis, float& splitPos);

		BVHNode* bvhNode = 0;
		Tri* tri = 0;
		uint* triIdx = 0;
		uint nodesUsed, triCount;
		mat4 invTransform;
		mat4 transform;
		aabb bounds; // <--- THIS must be present to fix the "bounds undefined" error
	};

	// Reference to a BLAS subtree (paper calls this a BRef)
	struct BRef
	{
		uint objectID = 0; // which BLAS / object this belongs to
		uint nodeIdx = 0;  // BLAS node index within that object
		float3 aabbMin = float3(1e30f);
		float3 aabbMax = float3(-1e30f);
		float3 centroid = float3(0);
		uint numPrims = 0; // SAH weight (approx ok)
	};

	// top-level node
	struct TLASNode
	{
		float3 aabbMin;
		uint leftRight; // Internal: Left Child Index. Leaf: BLAS Index.
		float3 aabbMax;
		uint BLASNode;  // Internal: 0. Leaf: BLAS Node Index + 1.

		bool isLeaf() { return BLASNode > 0; }
	};


	// top-level BVH
	class TLAS
	{
	public:
		TLAS() = default;
		TLAS(BVH* bvhList, int N);
		void Build();
		void Intersect(Ray& ray);
	private:
		void Subdivide(uint nodeIdx, std::vector<BRef>& segment, bool allowOpening);
		TLASNode* tlasNode = 0;
		BVH* blas = 0;
		uint nodesUsed, blasCount;
	};

	// game class
	class TopLevelApp : public TheApp
	{
	public:
		// game flow methods
		void Init();
		void Tick(float deltaTime);
		void Shutdown() {}
		// input handling
		void MouseUp(int button) {}
		void MouseDown(int button) {}
		void MouseMove(int x, int y) { mousePos.x = x, mousePos.y = y; }
		void MouseWheel(float y) {}
		void KeyUp(int key) {}
		void KeyDown(int key) {}
		// data members
		int2 mousePos;
		BVH bvh[64];
		TLAS tlas;
	};

} // namespace Tmpl8