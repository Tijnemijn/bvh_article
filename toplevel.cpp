#include "precomp.h"
#include "toplevel.h"

#include <algorithm>
#include <vector>

TheApp* CreateApp() { return new TopLevelApp(); }

//Intersection Functions

void IntersectTri(Ray& ray, const Tri& tri)
{
	ray.triTests++;
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
	const_cast<Ray&>(ray).aabbTests++;
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
	const_cast<Ray&>(ray).aabbTests++;
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

	//calculate rD and rD4 specifically for Object Space
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

static constexpr uint TLAS_NODE_CAPACITY = 8192;
static constexpr uint MAX_BREFS = 1024; // safety cap for opening phase

TLAS::TLAS(BVH* bvhList, int N)
{
	blas = bvhList;
	blasCount = N;
	// Allocate a generous node pool; partial re-braiding can increase leaf count.
	// Note: traversal assumes right child == left child + 1.
	tlasNode = (TLASNode*)_aligned_malloc(sizeof(TLASNode) * TLAS_NODE_CAPACITY, 64);
	nodesUsed = 2;
}

void TLAS::Build()
{
	// Partial re-braiding (paper): build TLAS over a list of BRefs.
	// Each segment: (optional) opening phase -> weighted SAH binning -> recurse.
	auto ComputeWorldBounds = [&](uint objectID, uint nodeIdx, float3& outMin, float3& outMax)
		{
			BVHNode& n = *blas[objectID].GetNode(nodeIdx);
			mat4& T = blas[objectID].GetTransform();
			aabb b;
			float3 bmin = n.aabbMin, bmax = n.aabbMax;
			for (int k = 0; k < 8; k++)
				b.grow(TransformPosition(float3(k & 1 ? bmax.x : bmin.x,
					k & 2 ? bmax.y : bmin.y, k & 4 ? bmax.z : bmin.z), T));
			outMin = b.bmin;
			outMax = b.bmax;
		};

	auto MakeBRef = [&](uint objectID, uint nodeIdx, uint numPrims) -> BRef
		{
			BRef r;
			r.objectID = objectID;
			r.nodeIdx = nodeIdx;
			r.numPrims = max(1u, numPrims);
			ComputeWorldBounds(objectID, nodeIdx, r.aabbMin, r.aabbMax);
			r.centroid = (r.aabbMin + r.aabbMax) * 0.5f;
			return r;
		};

	// Seed: one root BRef per object.
	std::vector<BRef> refs;
	refs.reserve(blasCount);
	for (uint i = 0; i < blasCount; i++)
		refs.push_back(MakeBRef(i, 0, blas[i].GetTriCount()));

	// Reset TLAS node pool.
	nodesUsed = 1;
	memset(tlasNode, 0, sizeof(TLASNode) * TLAS_NODE_CAPACITY);

	// Build recursively.
	Subdivide(0, refs, true);
}

static inline float Volume(const float3 bmin, const float3 bmax)
{
	float3 e = bmax - bmin;
	e.x = max(0.0f, e.x);
	e.y = max(0.0f, e.y);
	e.z = max(0.0f, e.z);
	return e.x * e.y * e.z;
}

static inline float GetAxis(const float3& v, const int axis)
{
	return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

static inline float IntersectionOverUnion(const float3 aMin, const float3 aMax, const float3 bMin, const float3 bMax)
{
	float3 iMin = fmaxf(aMin, bMin);
	float3 iMax = fminf(aMax, bMax);
	float iVol = Volume(iMin, iMax);
	if (iVol <= 0) return 0;
	float uVol = Volume(aMin, aMax) + Volume(bMin, bMax) - iVol;
	return uVol > 0 ? (iVol / uVol) : 0;
}

void TLAS::Subdivide(uint nodeIdx, std::vector<BRef>& segment, bool allowOpening)
{
	TLASNode& node = tlasNode[nodeIdx];
	// Segment bounds (also stored on this node).
	node.aabbMin = float3(1e30f);
	node.aabbMax = float3(-1e30f);
	for (const BRef& r : segment)
	{
		node.aabbMin = fminf(node.aabbMin, r.aabbMin);
		node.aabbMax = fmaxf(node.aabbMax, r.aabbMax);
	}

	// Leaf: store BRef (objectID + BLAS node).
	if (segment.size() == 1)
	{
		node.leftRight = segment[0].objectID;
		node.BLASNode = segment[0].nodeIdx + 1; // +1 to mark as Leaf
		return;
	}

	// Termination A: if all BRefs are from the same object, stop opening here and below.
	if (allowOpening)
	{
		bool allSameObject = true;
		const uint obj0 = segment[0].objectID;
		for (size_t i = 1; i < segment.size(); i++) if (segment[i].objectID != obj0) { allSameObject = false; break; }
		if (allSameObject) allowOpening = false;
	}

	// Termination B: for small segments, if overlap is tiny, stop opening.
	if (allowOpening && segment.size() <= 4)
	{
		float maxIou = 0.0f;
		for (size_t i = 0; i < segment.size(); i++)
			for (size_t j = i + 1; j < segment.size(); j++)
				maxIou = max(maxIou, IntersectionOverUnion(segment[i].aabbMin, segment[i].aabbMax, segment[j].aabbMin, segment[j].aabbMax));
		const float IOU_EARLY_OUT = 0.01f; // "very small" overlap
		if (maxIou < IOU_EARLY_OUT) allowOpening = false;
	}

	//  BRefs that are "wide" along the segment's longest axis.
	if (allowOpening)
	{
		float3 extent = node.aabbMax - node.aabbMin;
		int dim = 0;
		if (extent.y > extent.x) dim = 1;
		if (extent.z > extent[dim]) dim = 2;
		const float ext = extent[dim];
		if (ext > 0)
		{
			const float OPEN_THRESHOLD = 0.1f * ext;
			for (size_t i = 0; i < segment.size(); i++)
			{
				if (segment.size() >= MAX_BREFS) break; // safety cap
				BRef& r = segment[i];
				BVHNode* n = blas[r.objectID].GetNode(r.nodeIdx);
				if (n->isLeaf()) continue;
				const float width = r.aabbMax[dim] - r.aabbMin[dim];
				if (width <= OPEN_THRESHOLD) continue;

				//replace current by first child, append the other.
				const uint leftIdx = n->leftFirst;
				const uint rightIdx = n->leftFirst + 1;
				const uint leftPrims = max(1u, r.numPrims / 2);
				const uint rightPrims = max(1u, r.numPrims - leftPrims);
				BRef left = r;
				left.nodeIdx = leftIdx;
				left.numPrims = leftPrims;
				{
					float3 mn, mx;
					BVHNode& cn = *blas[left.objectID].GetNode(left.nodeIdx);
					mat4& T = blas[left.objectID].GetTransform();
					aabb b;
					float3 bmin = cn.aabbMin, bmax = cn.aabbMax;
					for (int k = 0; k < 8; k++)
						b.grow(TransformPosition(float3(k & 1 ? bmax.x : bmin.x,
							k & 2 ? bmax.y : bmin.y, k & 4 ? bmax.z : bmin.z), T));
					mn = b.bmin; mx = b.bmax;
					left.aabbMin = mn; left.aabbMax = mx;
					left.centroid = (mn + mx) * 0.5f;
				}

				BRef right = r;
				right.nodeIdx = rightIdx;
				right.numPrims = rightPrims;
				{
					float3 mn, mx;
					BVHNode& cn = *blas[right.objectID].GetNode(right.nodeIdx);
					mat4& T = blas[right.objectID].GetTransform();
					aabb b;
					float3 bmin = cn.aabbMin, bmax = cn.aabbMax;
					for (int k = 0; k < 8; k++)
						b.grow(TransformPosition(float3(k & 1 ? bmax.x : bmin.x,
							k & 2 ? bmax.y : bmin.y, k & 4 ? bmax.z : bmin.z), T));
					mn = b.bmin; mx = b.bmax;
					right.aabbMin = mn; right.aabbMax = mx;
					right.centroid = (mn + mx) * 0.5f;
				}

				r = left;
				segment.push_back(right);
			}
		}
	}

	// Weighted SAH binning split (paper): weight is numPrims per BRef.
	int bestAxis = -1;
	int bestSplitBin = -1;
	float bestCost = 1e30f;

	for (int a = 0; a < 3; a++)
	{
		float cmin = 1e30f, cmax = -1e30f;
		for (const BRef& r : segment)
		{
			cmin = min(cmin, GetAxis(r.centroid, a));
			cmax = max(cmax, GetAxis(r.centroid, a));
		}
		if (cmin == cmax) continue;

		struct Bin { aabb bounds; uint weight = 0; } bin[BINS];
		const float scale = (float)BINS / (cmax - cmin);
		for (const BRef& r : segment)
		{
			int b = min(BINS - 1, (int)((GetAxis(r.centroid, a) - cmin) * scale));
			bin[b].weight += r.numPrims;
			bin[b].bounds.grow(r.aabbMin);
			bin[b].bounds.grow(r.aabbMax);
		}

		aabb leftBox[BINS - 1], rightBox[BINS - 1];
		uint leftW[BINS - 1] = { 0 }, rightW[BINS - 1] = { 0 };
		aabb bL, bR;
		uint wL = 0, wR = 0;
		for (int i = 0; i < BINS - 1; i++)
		{
			wL += bin[i].weight;
			leftW[i] = wL;
			bL.grow(bin[i].bounds);
			leftBox[i] = bL;
			wR += bin[BINS - 1 - i].weight;
			rightW[BINS - 2 - i] = wR;
			bR.grow(bin[BINS - 1 - i].bounds);
			rightBox[BINS - 2 - i] = bR;
		}

		for (int i = 0; i < BINS - 1; i++)
		{
			if (leftW[i] == 0 || rightW[i] == 0) continue;
			float cost = leftBox[i].area() * (float)leftW[i] + rightBox[i].area() * (float)rightW[i];
			if (cost < bestCost)
			{
				bestCost = cost;
				bestAxis = a;
				bestSplitBin = i + 1;
			}
		}
	}

	std::vector<BRef> leftSeg, rightSeg;
	leftSeg.reserve(segment.size());
	rightSeg.reserve(segment.size());

	if (bestAxis != -1)
	{
		float cmin = 1e30f, cmax = -1e30f;
		for (const BRef& r : segment) { cmin = min(cmin, GetAxis(r.centroid, bestAxis)); cmax = max(cmax, GetAxis(r.centroid, bestAxis)); }
		float scale = (float)BINS / (cmax - cmin);
		for (const BRef& r : segment)
		{
			int b = min(BINS - 1, (int)((GetAxis(r.centroid, bestAxis) - cmin) * scale));
			if (b < bestSplitBin) leftSeg.push_back(r); else rightSeg.push_back(r);
		}
	}

	// Fallback: split by median centroid along longest axis.
	if (leftSeg.empty() || rightSeg.empty())
	{
		float3 extent = node.aabbMax - node.aabbMin;
		int dim = 0;
		if (extent.y > extent.x) dim = 1;
		if (extent.z > extent[dim]) dim = 2;
		std::sort(segment.begin(), segment.end(), [&](const BRef& a, const BRef& b) { return GetAxis(a.centroid, dim) < GetAxis(b.centroid, dim); });
		size_t mid = segment.size() / 2;
		leftSeg.assign(segment.begin(), segment.begin() + mid);
		rightSeg.assign(segment.begin() + mid, segment.end());
	}

	// Create children and recurse.
	const uint leftChildIdx = nodesUsed++;
	const uint rightChildIdx = nodesUsed++;
	node.leftRight = leftChildIdx;
	node.BLASNode = 0; // Internal

	TLASNode& lNode = tlasNode[leftChildIdx];
	TLASNode& rNode = tlasNode[rightChildIdx];
	lNode.aabbMin = float3(1e30f); lNode.aabbMax = float3(-1e30f);
	rNode.aabbMin = float3(1e30f); rNode.aabbMax = float3(-1e30f);
	for (const BRef& r : leftSeg) { lNode.aabbMin = fminf(lNode.aabbMin, r.aabbMin); lNode.aabbMax = fmaxf(lNode.aabbMax, r.aabbMax); }
	for (const BRef& r : rightSeg) { rNode.aabbMin = fminf(rNode.aabbMin, r.aabbMin); rNode.aabbMax = fmaxf(rNode.aabbMax, r.aabbMax); }

	Subdivide(leftChildIdx, leftSeg, allowOpening);
	Subdivide(rightChildIdx, rightSeg, allowOpening);
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

//App

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

	// --- METRIC: Build Time ---
	Timer buildTimer;
	tlas.Build();
	float buildTimeMs = buildTimer.elapsed() * 1000.0f;

	// --- METRIC: Build Space ---
	// Calculate BLAS Size (Nodes + Triangles + Indices)
	// Note: You might need to make 'nodesUsed' public in BVH/TLAS classes or add a GetSize() helper.
	// Assuming public access for this snippet:
	size_t blasSize = 0;
	for (int i = 0; i < 2; i++) { // Assuming 2 BLAS
		blasSize += bvh[i].nodesUsed * sizeof(BVHNode);
		blasSize += bvh[i].triCount * sizeof(Tri);
		blasSize += bvh[i].triCount * sizeof(uint);
	}

	// Calculate TLAS Size
	size_t tlasSize = tlas.nodesUsed * sizeof(TLASNode);
	size_t totalBytes = blasSize + tlasSize;

	unsigned long long frameTriTests = 0;
	unsigned long long frameNodeTests = 0;


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
	float frameTimeMs = t.elapsed() * 1000.0f;
	float currentFPS = 1000.0f / frameTimeMs;

	float secondsThisFrame = frameTimeMs / 1000.0f;
	totalTimeRecorded += secondsThisFrame;
	programDuration += secondsThisFrame;

	// --- CONTINUOUS CSV LOGGING (Every 60 Seconds) ---

	// 1. Gather data for this frame
	statsHistory.push_back({
		currentFPS,
		frameTriTests + frameNodeTests,
		buildTimeMs,
		totalBytes
		});
	totalTimeRecorded += (frameTimeMs / 1000.0f);

	// 2. If 60 seconds have passed, calculate averages and write to file
	if (totalTimeRecorded >= 60.0f)
	{
		// A. Calculate Averages
		double sumFPS = 0, sumInt = 0, sumBuild = 0, sumSize = 0;
		for (const auto& s : statsHistory) {
			sumFPS += s.fps;
			sumInt += s.totalIntersections;
			sumBuild += s.buildTime;
			sumSize += (double)s.sizeBytes;
		}
		double meanFPS = sumFPS / statsHistory.size();
		double meanInt = sumInt / statsHistory.size();
		double meanBuild = sumBuild / statsHistory.size();
		double meanSizeKB = (sumSize / statsHistory.size()) / 1024.0;

		// B. Calculate Standard Deviations
		double varFPS = 0, varInt = 0, varSize = 0;
		for (const auto& s : statsHistory) {
			varFPS += (s.fps - meanFPS) * (s.fps - meanFPS);
			varInt += (double(s.totalIntersections) - meanInt) * (double(s.totalIntersections) - meanInt);
			double sizeKB = (double)s.sizeBytes / 1024.0;
			varSize += (sizeKB - meanSizeKB) * (sizeKB - meanSizeKB);
		}
		double stdFPS = sqrt(varFPS / statsHistory.size());
		double stdInt = sqrt(varInt / statsHistory.size());
		double stdSize = sqrt(varSize / statsHistory.size());

		// C. Open CSV file (Append Mode)
		bool fileExists = std::ifstream("stats.csv").good();
		std::ofstream csvFile;
		csvFile.open("stats.csv", std::ios::out | std::ios::app);

		if (csvFile.is_open())
		{
			// Write Header if file is new (Added Timestamp column)
			if (!fileExists) {
				csvFile << "Timestamp,AvgFPS,StdDevFPS,AvgIntersections,StdDevIntersections,AvgBuildTimeMs,AvgMemoryKB,StdDevMemoryKB\n";
			}

			// Write Data Row
			csvFile << programDuration << "," // NEW: Current program time
				<< meanFPS << ","
				<< stdFPS << ","
				<< meanInt << ","
				<< stdInt << ","
				<< meanBuild << ","
				<< meanSizeKB << ","
				<< stdSize << "\n";

			csvFile.close();
			printf("\n[Stats Saved @ %.0fs] FPS: %.2f | Mem: %.2f KB\n", programDuration, meanFPS, meanSizeKB);
		}

		// D. Reset (programDuration is NOT reset)
		statsHistory.clear();
		totalTimeRecorded = 0.0f;
	}
}