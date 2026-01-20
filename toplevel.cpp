#include "precomp.h"
#include "toplevel.h"

TheApp* CreateApp() { return new TopLevelApp(); }

// --- Intersection Functions ---

void IntersectTri(Ray& ray, const Tri& tri)
{
	const float3 edge1 = tri.vertex1 - tri.vertex0;
	const float3 edge2 = tri.vertex2 - tri.vertex0;
	const float3 h = cross(ray.D, edge2);
	const float a = dot(edge1, h);
	if (a > -0.00001f && a < 0.00001f) return; // ray parallel to triangle
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

float IntersectAABB_SSE(const Ray& ray, const __m128& bmin4, const __m128& bmax4)
{
	static __m128 mask4 = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_set_ps(1, 0, 0, 0));
	__m128 t1 = _mm_mul_ps(_mm_sub_ps(_mm_and_ps(bmin4, mask4), ray.O4), ray.rD4);
	__m128 t2 = _mm_mul_ps(_mm_sub_ps(_mm_and_ps(bmax4, mask4), ray.O4), ray.rD4);
	__m128 vmax4 = _mm_max_ps(t1, t2), vmin4 = _mm_min_ps(t1, t2);
	float tmax = min(vmax4.m128_f32[0], min(vmax4.m128_f32[1], vmax4.m128_f32[2]));
	float tmin = max(vmin4.m128_f32[0], max(vmin4.m128_f32[1], vmin4.m128_f32[2]));
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

void BVH::SetTransform(mat4& newTransform)
{
	transform = newTransform;
	invTransform = newTransform.Inverted();
	// Calculate world-space bounds
	float3 bmin = bvhNode[0].aabbMin, bmax = bvhNode[0].aabbMax;
	bounds = aabb();
	for (int i = 0; i < 8; i++)
		bounds.grow(TransformPosition(float3(i & 1 ? bmax.x : bmin.x,
			i & 2 ? bmax.y : bmin.y, i & 4 ? bmax.z : bmin.z), newTransform));
}

void BVH::Intersect(Ray& ray, uint nodeIdx)
{
	// 1. Backup World Ray
	Ray backupRay = ray;

	// 2. Transform ray to Object Space
	ray.O = TransformPosition(ray.O, invTransform);
	ray.D = TransformVector(ray.D, invTransform);

	// FIX: Recalculate rD and rD4 specifically for Object Space
	// This was likely missing or incorrect, causing AABB tests to fail.
	ray.rD = float3(1.0f / ray.D.x, 1.0f / ray.D.y, 1.0f / ray.D.z);
	ray.rD4 = _mm_set_ps(0.0f, 1.0f / ray.D.z, 1.0f / ray.D.y, 1.0f / ray.D.x);

	// 3. Traversal
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
#ifdef USE_SSE
		float dist1 = IntersectAABB_SSE(ray, child1->aabbMin4, child1->aabbMax4);
		float dist2 = IntersectAABB_SSE(ray, child2->aabbMin4, child2->aabbMax4);
#else
		float dist1 = IntersectAABB(ray, child1->aabbMin, child1->aabbMax);
		float dist2 = IntersectAABB(ray, child2->aabbMin, child2->aabbMax);
#endif
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

	// 4. Restore Ray (keep closest t)
	backupRay.t = ray.t;
	ray = backupRay;
}

void BVH::Refit()
{
	// Refit logic (same as before)
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
	int axis;
	float splitPos;
	float splitCost = FindBestSplitPlane(node, axis, splitPos);
	float nosplitCost = node.CalculateNodeCost();
	if (splitCost >= nosplitCost) return;
	int i = node.leftFirst;
	int j = i + node.triCount - 1;
	while (i <= j)
	{
		if (tri[triIdx[i]].centroid[axis] < splitPos)
			i++;
		else
			swap(triIdx[i], triIdx[j--]);
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
	// Allocate 512 nodes (enough for re-braiding 256 items + overhead)
	tlasNode = (TLASNode*)_aligned_malloc(sizeof(TLASNode) * 512, 64);
	nodesUsed = 2;
}

void TLAS::Build()
{
	const int MAX_BUILD_ITEMS = 256;
	TLASBuildEntry buildItem[MAX_BUILD_ITEMS];
	int count = 0;

	// Initial pass: Add BLAS roots
	for (uint i = 0; i < blasCount; i++)
	{
		buildItem[count].blasIdx = i;
		buildItem[count].nodeIdx = 0;
		BVHNode& root = *blas[i].GetNode(0);
		mat4& T = blas[i].GetTransform();
		buildItem[count].bounds = aabb();
		float3 bmin = root.aabbMin, bmax = root.aabbMax;
		for (int k = 0; k < 8; k++)
			buildItem[count].bounds.grow(TransformPosition(float3(k & 1 ? bmax.x : bmin.x,
				k & 2 ? bmax.y : bmin.y, k & 4 ? bmax.z : bmin.z), T));
		buildItem[count].centroid = (buildItem[count].bounds.bmin + buildItem[count].bounds.bmax) * 0.5f;
		count++;
	}

	// Partial Re-braiding Loop
	while (count < MAX_BUILD_ITEMS - 1)
	{
		int bestIdx = -1;
		float maxArea = -1.0f;
		for (int i = 0; i < count; i++)
		{
			float area = buildItem[i].bounds.area();
			if (area > maxArea) { maxArea = area; bestIdx = i; }
		}
		if (bestIdx == -1) break;

		BVH& b = blas[buildItem[bestIdx].blasIdx];
		BVHNode* node = b.GetNode(buildItem[bestIdx].nodeIdx);

		if (node->isLeaf()) { break; }

		uint leftIdx = node->leftFirst;
		uint rightIdx = node->leftFirst + 1;
		BVHNode* left = b.GetNode(leftIdx);
		BVHNode* right = b.GetNode(rightIdx);
		mat4& T = b.GetTransform();

		TLASBuildEntry oldItem = buildItem[bestIdx];
		buildItem[bestIdx] = buildItem[count - 1];
		count--;

		auto AddChild = [&](BVHNode* child, uint idx) {
			TLASBuildEntry& entry = buildItem[count++];
			entry.blasIdx = oldItem.blasIdx;
			entry.nodeIdx = idx;
			entry.bounds = aabb();
			float3 bmin = child->aabbMin, bmax = child->aabbMax;
			for (int k = 0; k < 8; k++)
				entry.bounds.grow(TransformPosition(float3(k & 1 ? bmax.x : bmin.x,
					k & 2 ? bmax.y : bmin.y, k & 4 ? bmax.z : bmin.z), T));
			entry.centroid = (entry.bounds.bmin + entry.bounds.bmax) * 0.5f;
			};

		AddChild(left, leftIdx);
		AddChild(right, rightIdx);
	}

	// Prepare for Subdivide
	nodesUsed = 1; // Start filling from index 1 (0 is root)

	// Clear memory to prevent ghost data
	memset(tlasNode, 0, sizeof(TLASNode) * 512);

	uint* indices = new uint[count];
	for (int i = 0; i < count; i++) indices[i] = i;

	// Set Root Bounds (Node 0) explicitly before subdivision
	TLASNode& root = tlasNode[0];
	root.aabbMin = float3(1e30f); root.aabbMax = float3(-1e30f);
	for (int i = 0; i < count; i++) {
		root.aabbMin = fminf(root.aabbMin, buildItem[i].bounds.bmin);
		root.aabbMax = fmaxf(root.aabbMax, buildItem[i].bounds.bmax);
	}

	Subdivide(0, buildItem, indices, count);
	delete[] indices;
}

void TLAS::Subdivide(uint nodeIdx, TLASBuildEntry* buildList, uint* indices, int count)
{
	TLASNode& node = tlasNode[nodeIdx];
	if (count == 1)
	{
		uint itemIdx = indices[0];
		node.leftRight = buildList[itemIdx].blasIdx;
		node.BLASNode = buildList[itemIdx].nodeIdx + 1; // +1 to mark as Leaf
		return;
	}

	int bestAxis = -1;
	float bestSplitPos = 0;
	float bestCost = 1e30f;

	// Simple SAH (Sweep)
	for (int a = 0; a < 3; a++)
	{
		// Sort by centroid
		for (int i = 0; i < count - 1; i++) for (int j = i + 1; j < count; j++)
			if (buildList[indices[i]].centroid[a] > buildList[indices[j]].centroid[a])
				swap(indices[i], indices[j]);

		float* rightAreas = new float[count];
		aabb boxR;
		for (int i = count - 1; i > 0; i--) {
			boxR.grow(buildList[indices[i]].bounds);
			rightAreas[i] = boxR.area();
		}

		aabb boxL;
		for (int i = 0; i < count - 1; i++) {
			boxL.grow(buildList[indices[i]].bounds);
			float cost = (i + 1) * boxL.area() + (count - 1 - i) * rightAreas[i + 1];
			if (cost < bestCost) {
				bestCost = cost;
				bestAxis = a;
				bestSplitPos = (float)(i + 1);
			}
		}
		delete[] rightAreas;
	}

	// Fallback split
	if (bestAxis == -1) { bestAxis = 0; bestSplitPos = count / 2.0f; }

	// Ensure list is sorted by best axis one last time
	for (int i = 0; i < count - 1; i++) for (int j = i + 1; j < count; j++)
		if (buildList[indices[i]].centroid[bestAxis] > buildList[indices[j]].centroid[bestAxis])
			swap(indices[i], indices[j]);

	int leftCount = (int)bestSplitPos;
	int leftChildIdx = nodesUsed++;
	int rightChildIdx = nodesUsed++;

	node.leftRight = leftChildIdx;
	node.BLASNode = 0; // Internal

	// Calculate bounds for children
	TLASNode& lNode = tlasNode[leftChildIdx];
	TLASNode& rNode = tlasNode[rightChildIdx];
	lNode.aabbMin = float3(1e30f); lNode.aabbMax = float3(-1e30f);
	rNode.aabbMin = float3(1e30f); rNode.aabbMax = float3(-1e30f);

	for (int i = 0; i < leftCount; i++) {
		lNode.aabbMin = fminf(lNode.aabbMin, buildList[indices[i]].bounds.bmin);
		lNode.aabbMax = fmaxf(lNode.aabbMax, buildList[indices[i]].bounds.bmax);
	}
	for (int i = leftCount; i < count; i++) {
		rNode.aabbMin = fminf(rNode.aabbMin, buildList[indices[i]].bounds.bmin);
		rNode.aabbMax = fmaxf(rNode.aabbMax, buildList[indices[i]].bounds.bmax);
	}

	Subdivide(leftChildIdx, buildList, indices, leftCount);
	Subdivide(rightChildIdx, buildList, indices + leftCount, count - leftCount);
}

void TLAS::Intersect(Ray& ray)
{
	TLASNode* node = &tlasNode[0], * stack[64];
	uint stackPtr = 0;
	while (1)
	{
		if (node->isLeaf())
		{
			blas[node->leftRight].Intersect(ray, node->BLASNode - 1);
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		TLASNode* child1 = &tlasNode[node->leftRight];
		TLASNode* child2 = &tlasNode[node->leftRight + 1];

		// Use SSE intersection for TLAS as well if possible, or robust standard
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

// --- App ---

void TopLevelApp::Init()
{
	bvh[0] = BVH("assets/armadillo.tri", 30000);
	bvh[1] = BVH("assets/armadillo.tri", 30000);
	tlas = TLAS(bvh, 2);
}

void TopLevelApp::Tick(float deltaTime)
{
	float3 p0(-1, 1, 2), p1(1, 1, 2), p2(-1, -1, 2);
	static float angle = 0;
	angle += 0.01f;
	if (angle > 2 * PI) angle -= 2 * PI;

	bvh[0].SetTransform(mat4::Translate(float3(-1.3f, 0, 0)));
	bvh[1].SetTransform(mat4::Translate(float3(1.3f, 0, 0)) * mat4::RotateY(angle));

	tlas.Build();

	Timer t;

#pragma omp parallel for schedule(dynamic)
	for (int tile = 0; tile < (SCRWIDTH * SCRHEIGHT / 64); tile++)
	{
		int x = tile % (SCRWIDTH / 8), y = tile / (SCRWIDTH / 8);
		Ray ray;
		ray.O = float3(0, 0.5f, -4.5f);
		for (int v = 0; v < 8; v++) for (int u = 0; u < 8; u++)
		{
			float3 pixelPos = ray.O + p0 +
				(p1 - p0) * ((x * 8 + u) / (float)SCRWIDTH) +
				(p2 - p0) * ((y * 8 + v) / (float)SCRHEIGHT);
			ray.D = normalize(pixelPos - ray.O), ray.t = 1e30f;

			// Init SSE part of ray direction for TLAS usage
			ray.rD = float3(1.0f / ray.D.x, 1.0f / ray.D.y, 1.0f / ray.D.z);
			ray.rD4 = _mm_set_ps(0.0f, ray.rD.z, ray.rD.y, ray.rD.x);

			tlas.Intersect(ray);
			uint c = ray.t < 1e30f ? (255 - (int)((ray.t - 3) * 80)) : 0;
			screen->Plot(x * 8 + u, y * 8 + v, c * 0x10101);
		}
	}
	float elapsed = t.elapsed() * 1000;
	printf("tracing time: %.2fms (%5.2fK rays/s)\n", elapsed, sqr(630) / elapsed);
}