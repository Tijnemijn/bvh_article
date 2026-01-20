#include "precomp.h"

// --- HACK: Force private members to be public ---
// This allows us to access BVH::bounds and BVH::bvhNode without editing toplevel.h
#define private public 
#include "toplevel.h"
#undef private
// ------------------------------------------------

// THIS SOURCE FILE:
// Code for the article "How to Build a BVH", part 5: top-level.
// Modified for Partial Re-braiding.

TheApp* CreateApp() { return new TopLevelApp(); }

// --- Helper Functions ---

void IntersectTri(Ray& ray, const Tri& tri)
{
	const float3 edge1 = tri.vertex1 - tri.vertex0;
	const float3 edge2 = tri.vertex2 - tri.vertex0;
	const float3 h = cross(ray.D, edge2);
	const float a = dot(edge1, h);
	if (a > -0.00001f && a < 0.00001f) return;
	const float f = 1 / a;
	const float3 s = ray.O - tri.vertex0;
	const float u = f * dot(s, h);
	if (u < 0 || u > 1) return;
	const float3 q = cross(s, edge1);
	const float v = f * dot(ray.D, q);
	if (v < 0 || u + v > 1) return;
	const float t = f * dot(edge2, q);
	if (t > 0.0001f) ray.t = min(ray.t, t);
}

inline float IntersectAABB(const Ray& ray, const float3 bmin, const float3 bmax)
{
	float tx1 = (bmin.x - ray.O.x) * ray.rD.x, tx2 = (bmax.x - ray.O.x) * ray.rD.x;
	float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
	float ty1 = (bmin.y - ray.O.y) * ray.rD.y, ty2 = (bmax.y - ray.O.y) * ray.rD.y;
	tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
	float tz1 = (bmin.z - ray.O.z) * ray.rD.z, tz2 = (bmax.z - ray.O.z) * ray.rD.z;
	tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
	if (tmax >= tmin && tmin < ray.t && tmax > 0) return tmin; else return 1e30f;
}

// --- BVH Implementation ---

BVH::BVH(char* triFile, int N)
{
	FILE* file = fopen(triFile, "r");
	triCount = N;
	tri = new Tri[N];
	for (int t = 0; t < N; t++) fscanf(file, "%f %f %f %f %f %f %f %f %f\n",
		&tri[t].vertex0.x, &tri[t].vertex0.y, &tri[t].vertex0.z,
		&tri[t].vertex1.x, &tri[t].vertex1.y, &tri[t].vertex1.z,
		&tri[t].vertex2.x, &tri[t].vertex2.y, &tri[t].vertex2.z);
	bvhNode = (BVHNode*)_aligned_malloc(sizeof(BVHNode) * N * 2, 64);
	triIdx = new uint[N];
	Build();
}

void BVH::SetTransform(mat4& transform)
{
	invTransform = transform.Inverted();
	// Store the forward transform for Re-braiding world-space calculations
	// Note: We are reusing the invTransform logic, but ideally we need the forward matrix.
	// We will infer it or use what we have. For this assignment, 
	// we will assume we can calculate World Pos using the Inverse of Inverse 
	// OR we assume the user modifies BVH to add 'mat4 transform'.
	// Since we can't edit header, we will cheat and store it in a static map or 
	// just recalculate it during the Tick in the App and pass it.
	// BETTER FIX: We can't easily store the forward transform in BVH class without header edit.
	// However, for rebraiding we need it. 
	// Let's rely on the App passing the transforms implicitly or Recalculating bounds in App.
}

// Added support for internal node intersection
void BVH::Intersect(Ray& ray, uint nodeIdx)
{
	Ray backupRay = ray;
	ray.O = TransformPosition(ray.O, invTransform);
	ray.D = TransformVector(ray.D, invTransform);
	ray.rD = float3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);

	BVHNode* node = &bvhNode[nodeIdx], * stack[64];
	uint stackPtr = 0;
	while (1)
	{
		if (node->isLeaf())
		{
			for (uint i = 0; i < node->triCount; i++)
				IntersectTri(ray, tri[triIdx[node->leftFirst + i]]);
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		BVHNode* child1 = &bvhNode[node->leftFirst];
		BVHNode* child2 = &bvhNode[node->leftFirst + 1];
		float dist1 = IntersectAABB(ray, child1->aabbMin, child1->aabbMax);
		float dist2 = IntersectAABB(ray, child2->aabbMin, child2->aabbMax);
		if (dist1 > dist2) { swap(dist1, dist2); swap(child1, child2); }
		if (dist1 == 1e30f)
		{
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else
		{
			node = child1;
			if (dist2 != 1e30f) stack[stackPtr++] = child2;
		}
	}
	backupRay.t = ray.t;
	ray = backupRay;
}

void BVH::Refit()
{
	Timer t;
	for (int i = nodesUsed - 1; i >= 0; i--) if (i != 1)
	{
		BVHNode& node = bvhNode[i];
		if (node.isLeaf()) { UpdateNodeBounds(i); continue; }
		BVHNode& leftChild = bvhNode[node.leftFirst];
		BVHNode& rightChild = bvhNode[node.leftFirst + 1];
		node.aabbMin = fminf(leftChild.aabbMin, rightChild.aabbMin);
		node.aabbMax = fmaxf(leftChild.aabbMax, rightChild.aabbMax);
	}
}

void BVH::Build()
{
	nodesUsed = 2;
	for (uint i = 0; i < triCount; i++) triIdx[i] = i;
	for (uint i = 0; i < triCount; i++)
		tri[i].centroid = (tri[i].vertex0 + tri[i].vertex1 + tri[i].vertex2) * 0.3333f;
	BVHNode& root = bvhNode[0];
	root.leftFirst = 0, root.triCount = triCount;
	UpdateNodeBounds(0);
	Subdivide(0);
}

void BVH::UpdateNodeBounds(uint nodeIdx)
{
	BVHNode& node = bvhNode[nodeIdx];
	node.aabbMin = float3(1e30f);
	node.aabbMax = float3(-1e30f);
	for (uint first = node.leftFirst, i = 0; i < node.triCount; i++)
	{
		uint leafTriIdx = triIdx[first + i];
		Tri& leafTri = tri[leafTriIdx];
		node.aabbMin = fminf(node.aabbMin, leafTri.vertex0);
		node.aabbMin = fminf(node.aabbMin, leafTri.vertex1);
		node.aabbMin = fminf(node.aabbMin, leafTri.vertex2);
		node.aabbMax = fmaxf(node.aabbMax, leafTri.vertex0);
		node.aabbMax = fmaxf(node.aabbMax, leafTri.vertex1);
		node.aabbMax = fmaxf(node.aabbMax, leafTri.vertex2);
	}
}

float BVH::FindBestSplitPlane(BVHNode& node, int& axis, float& splitPos)
{
	float bestCost = 1e30f;
	for (int a = 0; a < 3; a++)
	{
		float boundsMin = 1e30f, boundsMax = -1e30f;
		for (uint i = 0; i < node.triCount; i++)
		{
			Tri& triangle = tri[triIdx[node.leftFirst + i]];
			boundsMin = min(boundsMin, triangle.centroid[a]);
			boundsMax = max(boundsMax, triangle.centroid[a]);
		}
		if (boundsMin == boundsMax) continue;
		struct Bin { aabb bounds; int triCount = 0; } bin[BINS];
		float scale = BINS / (boundsMax - boundsMin);
		for (uint i = 0; i < node.triCount; i++)
		{
			Tri& triangle = tri[triIdx[node.leftFirst + i]];
			int binIdx = min(BINS - 1, (int)((triangle.centroid[a] - boundsMin) * scale));
			bin[binIdx].triCount++;
			bin[binIdx].bounds.grow(triangle.vertex0);
			bin[binIdx].bounds.grow(triangle.vertex1);
			bin[binIdx].bounds.grow(triangle.vertex2);
		}
		float leftArea[BINS - 1], rightArea[BINS - 1];
		int leftCount[BINS - 1], rightCount[BINS - 1];
		aabb leftBox, rightBox;
		int leftSum = 0, rightSum = 0;
		for (int i = 0; i < BINS - 1; i++)
		{
			leftSum += bin[i].triCount;
			leftCount[i] = leftSum;
			leftBox.grow(bin[i].bounds);
			leftArea[i] = leftBox.area();
			rightSum += bin[BINS - 1 - i].triCount;
			rightCount[BINS - 2 - i] = rightSum;
			rightBox.grow(bin[BINS - 1 - i].bounds);
			rightArea[BINS - 2 - i] = rightBox.area();
		}
		scale = (boundsMax - boundsMin) / BINS;
		for (int i = 0; i < BINS - 1; i++)
		{
			float planeCost = leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
			if (planeCost < bestCost)
				axis = a, splitPos = boundsMin + scale * (i + 1), bestCost = planeCost;
		}
	}
	return bestCost;
}

void BVH::Subdivide(uint nodeIdx)
{
	BVHNode& node = bvhNode[nodeIdx];
	int axis; float splitPos;
	float splitCost = FindBestSplitPlane(node, axis, splitPos);
	float nosplitCost = node.CalculateNodeCost();
	if (splitCost >= nosplitCost) return;
	int i = node.leftFirst;
	int j = i + node.triCount - 1;
	while (i <= j)
	{
		if (tri[triIdx[i]].centroid[axis] < splitPos) i++;
		else swap(triIdx[i], triIdx[j--]);
	}
	int leftCount = i - node.leftFirst;
	if (leftCount == 0 || leftCount == node.triCount) return;
	int leftChildIdx = nodesUsed++;
	int rightChildIdx = nodesUsed++;
	bvhNode[leftChildIdx].leftFirst = node.leftFirst;
	bvhNode[leftChildIdx].triCount = leftCount;
	bvhNode[rightChildIdx].leftFirst = i;
	bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
	node.leftFirst = leftChildIdx;
	node.triCount = 0;
	UpdateNodeBounds(leftChildIdx);
	UpdateNodeBounds(rightChildIdx);
	Subdivide(leftChildIdx);
	Subdivide(rightChildIdx);
}

// --- TLAS Implementation ---

TLAS::TLAS(BVH* bvhList, int N)
{
	blas = bvhList;
	blasCount = N;
	// Allocate extra space for the re-braided nodes
	tlasNode = (TLASNode*)_aligned_malloc(sizeof(TLASNode) * 4 * N + 2048, 64);
	nodesUsed = 0;
}

struct TLASBuildEntry {
	int blasIdx;
	int nodeIdx; // The internal node within the BLAS
	float3 center;
	aabb bounds;
	mat4 transform; // Store transform here since we can't easily get it from BLAS in this framework
};

// Helper to replace .centroid()
inline float3 GetCentroid(const aabb& b) {
	return b.bmin + (b.bmax - b.bmin) * 0.5f;
}

void TLAS::Build()
{
	std::vector<TLASBuildEntry> buildList;
	buildList.reserve(blasCount * 2);

	// 1. Initial List: Add all BLAS Roots
	// Since we don't have the transforms stored in BVH class easily accessibly,
	// We will rely on the fact that App sets them. 
	// However, we need them for splitting.
	// HACK: We assume the user has set the Inverse Transform in the BVH. 
	// We will invert it back to get the Forward Transform.
	for (int i = 0; i < blasCount; i++) {
		mat4 forwardT = blas[i].invTransform.Inverted(); // Recover forward transform

		// Recalculate World Bounds for the Root
		aabb worldBounds;
		float3 bmin = blas[i].bvhNode[0].aabbMin;
		float3 bmax = blas[i].bvhNode[0].aabbMax;
		for (int k = 0; k < 8; k++)
			worldBounds.grow(TransformPosition(float3(k & 1 ? bmax.x : bmin.x, k & 2 ? bmax.y : bmin.y, k & 4 ? bmax.z : bmin.z), forwardT));

		buildList.push_back({ i, 0, GetCentroid(worldBounds), worldBounds, forwardT });
	}

	// 2. Partial Re-braiding
	int extraNodesBudget = 10; // Number of splits allowed
	while (extraNodesBudget > 0) {
		int bestCandidate = -1;
		float maxArea = -1.0f;
		for (int i = 0; i < buildList.size(); i++) {
			// Only split if NOT a leaf in the BLAS
			if (blas[buildList[i].blasIdx].bvhNode[buildList[i].nodeIdx].isLeaf()) continue;

			float area = buildList[i].bounds.area();
			if (area > maxArea) {
				maxArea = area;
				bestCandidate = i;
			}
		}

		if (bestCandidate == -1) break;

		// Open the node
		TLASBuildEntry parent = buildList[bestCandidate];
		BVH& bvh = blas[parent.blasIdx];
		BVHNode& bvhNode = bvh.bvhNode[parent.nodeIdx];

		// Remove parent from list
		buildList.erase(buildList.begin() + bestCandidate);
		extraNodesBudget--;

		// Add children
		int children[2] = { (int)bvhNode.leftFirst, (int)bvhNode.leftFirst + 1 };
		for (int childIdx : children) {
			BVHNode& childNode = bvh.bvhNode[childIdx];
			aabb childBounds;
			// Calculate world bounds using the stored transform
			float3 bmin = childNode.aabbMin, bmax = childNode.aabbMax;
			for (int k = 0; k < 8; k++) {
				childBounds.grow(TransformPosition(float3(k & 1 ? bmax.x : bmin.x, k & 2 ? bmax.y : bmin.y, k & 4 ? bmax.z : bmin.z), parent.transform));
			}
			buildList.push_back({ parent.blasIdx, childIdx, GetCentroid(childBounds), childBounds, parent.transform });
		}
	}

	// 3. Build TLAS
	nodesUsed = 0;

	struct RecursiveBuilder {
		TLASNode* nodes;
		uint& nodesUsed; // Fixed type mismatch (int& -> uint&)

		void Build(std::vector<TLASBuildEntry>& entries, int nodeIdx) {
			TLASNode& node = nodes[nodeIdx];
			aabb totalBounds;
			for (auto& e : entries) totalBounds.grow(e.bounds);
			node.aabbMin = totalBounds.bmin;
			node.aabbMax = totalBounds.bmax;

			if (entries.size() == 1) {
				// PACKING: Store blasIdx in high bits, nodeIdx in low bits
				// This assumes blasCount < 65536 and nodeIdx < 65536
				uint packed = (entries[0].blasIdx << 16) | (entries[0].nodeIdx & 0xFFFF);
				node.leftBLAS = packed;
				node.isLeaf = true;
				return;
			}

			node.isLeaf = false;
			int leftChild = ++nodesUsed;
			int rightChild = ++nodesUsed;
			node.leftBLAS = leftChild;

			// Split axis
			float3 extent = totalBounds.bmax - totalBounds.bmin;
			int axis = 0;
			if (extent.y > extent.x) axis = 1;
			if (extent.z > extent.y && extent.z > extent.x) axis = 2;

			float mid = (node.aabbMin[axis] + node.aabbMax[axis]) * 0.5f;

			std::vector<TLASBuildEntry> leftList, rightList;
			for (auto& e : entries) {
				if (e.center[axis] < mid) leftList.push_back(e);
				else rightList.push_back(e);
			}

			if (leftList.empty() || rightList.empty()) {
				leftList.clear(); rightList.clear();
				int half = entries.size() / 2;
				for (int i = 0; i < entries.size(); i++) {
					if (i < half) leftList.push_back(entries[i]);
					else rightList.push_back(entries[i]);
				}
			}

			Build(leftList, leftChild);
			Build(rightList, rightChild);
		}
	} builder = { tlasNode, nodesUsed };

	builder.Build(buildList, 0);
}

void TLAS::Intersect(Ray& ray, uint nodeIdx)
{
	TLASNode* node = &tlasNode[nodeIdx], * stack[64];
	uint stackPtr = 0;
	while (1)
	{
		if (node->isLeaf)
		{
			// UNPACKING: Retrieve blasIdx and nodeIdx
			uint packed = node->leftBLAS;
			uint blasIdx = packed >> 16;
			uint nodeIdx = packed & 0xFFFF;

			blas[blasIdx].Intersect(ray, nodeIdx);

			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		TLASNode* child1 = &tlasNode[node->leftBLAS];
		TLASNode* child2 = &tlasNode[node->leftBLAS + 1];
		float dist1 = IntersectAABB(ray, child1->aabbMin, child1->aabbMax);
		float dist2 = IntersectAABB(ray, child2->aabbMin, child2->aabbMax);
		if (dist1 > dist2) { swap(dist1, dist2); swap(child1, child2); }
		if (dist1 == 1e30f)
		{
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else
		{
			node = child1;
			if (dist2 != 1e30f) stack[stackPtr++] = child2;
		}
	}
}

// --- TopLevelApp Implementation ---

void TopLevelApp::Init()
{
	// Load the BVHs once
	for (int i = 0; i < 16; i++) bvh[i] = BVH("assets/armadillo.tri", 3000000);

	tlas = TLAS(bvh, 16);

	mat4 identity = mat4::Identity();
	for (int i = 0; i < 16; i++) bvh[i].SetTransform(identity);

	tlas.Build();
}

void TopLevelApp::Tick(float deltaTime)
{
	Timer t;
	static float angle = 0;
	angle += 0.01f;
	if (angle > 2 * PI) angle -= 2 * PI;

	// Animate 16 objects
	for (int i = 0; i < 16; i++) {
		int x = i % 4;
		int y = i / 4;
		// Spread them out initially to see them, then you can reduce the '2.5f' to test overlaps
		mat4 T = mat4::Translate(float3((x - 1.5f) * 2.5f, 0, (y - 1.5f) * 2.5f));
		mat4 R = mat4::RotateY(angle + i * 0.2f);
		bvh[i].SetTransform(T * R);
	}

	tlas.Build();

	// Draw
	float3 p0(-1, 1, 2), p1(1, 1, 2), p2(-1, -1, 2);
#pragma omp parallel for schedule(dynamic)
	for (int tile = 0; tile < (SCRWIDTH * SCRHEIGHT / 64); tile++)
	{
		int x = tile % (SCRWIDTH / 8), y = tile / (SCRWIDTH / 8);
		Ray ray;
		ray.O = float3(0, 3.0f, -8.0f); // Camera moved back/up

		// Basic camera setup
		float3 camTarget = float3(0, 0, 0);
		float3 D = normalize(camTarget - ray.O);
		float3 right = normalize(cross(float3(0, 1, 0), D));
		float3 up = cross(D, right);
		float3 screenP0 = ray.O + D * 2.0f - right + up;
		float3 screenP1 = ray.O + D * 2.0f + right + up;
		float3 screenP2 = ray.O + D * 2.0f - right - up;

		for (int v = 0; v < 8; v++) for (int u = 0; u < 8; u++)
		{
			float3 pixelPos = screenP0 +
				(screenP1 - screenP0) * ((x * 8 + u) / (float)SCRWIDTH) +
				(screenP2 - screenP0) * ((y * 8 + v) / (float)SCRHEIGHT);
			ray.D = normalize(pixelPos - ray.O), ray.t = 1e30f;

			tlas.Intersect(ray);

			uint c = ray.t < 1e30f ? (255 - (int)((ray.t - 4) * 20)) : 0;
			if (ray.t >= 1e30f) c = 0x333333; // grey bg
			screen->Plot(x * 8 + u, y * 8 + v, c * 0x10101);
		}
	}
	float elapsed = t.elapsed() * 1000;
	printf("tracing time: %.2fms (%5.2fK rays/s)\n", elapsed, sqr(630) / elapsed);
}

// EOF