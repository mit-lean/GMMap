RTREE_TEMPLATE
RTREE_QUAL::RTree()
{
    ASSERT(MAXNODES > MINNODES);
    ASSERT(MINNODES > 0);

    // Precomputed volumes of the unit spheres for the first few dimensions
    const float UNIT_SPHERE_VOLUMES[] = {
            0.000000f, 2.000000f, 3.141593f, // Dimension  0,1,2
            4.188790f, 4.934802f, 5.263789f, // Dimension  3,4,5
            5.167713f, 4.724766f, 4.058712f, // Dimension  6,7,8
            3.298509f, 2.550164f, 1.884104f, // Dimension  9,10,11
            1.335263f, 0.910629f, 0.599265f, // Dimension  12,13,14
            0.381443f, 0.235331f, 0.140981f, // Dimension  15,16,17
            0.082146f, 0.046622f, 0.025807f, // Dimension  18,19,20
    };

    m_root = AllocNode();
    m_root->m_level = 0;
    m_unitSphereVolume = (ELEMTYPEREAL)UNIT_SPHERE_VOLUMES[NUMDIMS];
}

RTREE_TEMPLATE
RTREE_QUAL::RTree(const RTree& other) : RTree()
{
    CopyRec(m_root, other.m_root);
}


RTREE_TEMPLATE
RTREE_QUAL::~RTree()
{
    Reset(); // Free, or reset node memory
}

RTREE_TEMPLATE
RTREE_QUAL& RTREE_QUAL::operator=(RTree tree)
{
    swap(*this, tree);
    return *this;
}

RTREE_TEMPLATE
RTREE_QUAL::RTree(RTree&& tree) noexcept : m_root(tree.m_root), m_unitSphereVolume(tree.m_unitSphereVolume)
{
    tree.m_root = NULL;
    tree.m_unitSphereVolume = (ELEMTYPEREAL) 0;
}

RTREE_TEMPLATE
void RTREE_QUAL::Insert(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], const DATATYPE& a_dataId)
{
    // Create node to be inserted
    Branch branch;
    branch.m_data = a_dataId;
    branch.m_child = NULL;
    branch.m_rect = Rect(a_min, a_max);

    // Insert into the RTree
    InsertRect(branch, &m_root, 0);
}

RTREE_TEMPLATE
void RTREE_QUAL::Insert(const Rect_Vec& lowerBound, const Rect_Vec& upperBound, const DATATYPE& a_dataId){
    // Create node to be inserted (using the eigen interface)
    Branch branch;
    branch.m_data = a_dataId;
    branch.m_child = NULL;
    branch.m_rect = Rect(lowerBound, upperBound);

    // Insert into the RTree
    InsertRect(branch, &m_root, 0);
}

RTREE_TEMPLATE
void RTREE_QUAL::Insert(const Rect& BBox, const DATATYPE& a_dataId){
    // Create node to be inserted (using the eigen interface)
    Branch branch;
    branch.m_data = a_dataId;
    branch.m_child = NULL;
    branch.m_rect = BBox;

    // Insert into the RTree
    InsertRect(branch, &m_root, 0);
}

RTREE_TEMPLATE
void RTREE_QUAL::Remove(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], const DATATYPE& a_dataId)
{
    Rect rect(a_min, a_max);
    RemoveRect(&rect, a_dataId, &m_root);
}

RTREE_TEMPLATE
void RTREE_QUAL::Remove(const Rect_Vec &lowerBound, const Rect_Vec &upperBound,
                        const DATATYPE &a_dataId) {
    Rect rect(lowerBound, upperBound);
    RemoveRect(&rect, a_dataId, &m_root);
}

RTREE_TEMPLATE
void RTREE_QUAL::Remove(const Rect& BBox,
                        const DATATYPE &a_dataId) {
    RemoveRect(&BBox, a_dataId, &m_root);
}

// Search with callback function support
RTREE_TEMPLATE
int RTREE_QUAL::Search(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], std::function<bool (const DATATYPE&)> callback) const
{
    Rect rect(a_min, a_max);
    // NOTE: May want to return search result another way, perhaps returning the number of found elements here.
    int foundCount = 0;
    Search(m_root, &rect, foundCount, callback);

    return foundCount;
}

// Search that returns a list of intersected ID
RTREE_TEMPLATE
std::vector<DATATYPE> RTREE_QUAL::Search(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS]) const
{
    Rect rect(a_min, a_max);

    // NOTE: May want to return search result another way, perhaps returning the number of found elements here.
    std::vector<DATATYPE> results;

    Search(m_root, &rect, results);

    return results;
}

RTREE_TEMPLATE
std::vector<DATATYPE> RTREE_QUAL::Search(const Rect_Vec &BBoxLow, const Rect_Vec &BBoxHigh) const {
    Rect rect(BBoxLow, BBoxHigh);

    // NOTE: May want to return search result another way, perhaps returning the number of found elements here.
    std::vector<DATATYPE> results;

    Search(m_root, &rect, results);

    return results;
}

RTREE_TEMPLATE
std::vector<DATATYPE> RTREE_QUAL::Search(const Rect& BBox) const {
    // NOTE: May want to return search result another way, perhaps returning the number of found elements here.
    std::vector<DATATYPE> results;

    Search(m_root, &BBox, results);

    return results;
}


RTREE_TEMPLATE
int RTREE_QUAL::CountLeafNodes() const
{
    int count = 0;
    CountLeafRec(m_root, count);

    return count;
}

RTREE_TEMPLATE
int RTREE_QUAL::CountAllNodes() const
{
    int count = 0;
    CountAllRec(m_root, count);
    return count;
}



RTREE_TEMPLATE
void RTREE_QUAL::CountLeafRec(Node* a_node, int& a_count) const
{
    if(a_node->IsInternalNode())  // not a leaf node
    {
        for(int index = 0; index < a_node->m_count; ++index)
        {
            CountLeafRec(a_node->m_branch[index].m_child, a_count);
        }
    }
    else // A leaf node
    {
        a_count += a_node->m_count;
    }
}

RTREE_TEMPLATE
void RTREE_QUAL::CountAllRec(Node* a_node, int& a_count) const
{
    if (a_node == m_root && a_node == NULL){
        return;
    }
    // Count the number of rectangles in the current node;
    a_count += a_node->m_count;
    if(a_node->IsInternalNode())  // not a leaf node
    {
        for(int index = 0; index < a_node->m_count; ++index)
        {
            // Count all rectangles in the children
            CountAllRec(a_node->m_branch[index].m_child, a_count);
        }
    }
}

RTREE_TEMPLATE
void RTREE_QUAL::RemoveAll()
{
    // Delete all existing nodes
    Reset();

    m_root = AllocNode();
    m_root->m_level = 0;
}

RTREE_TEMPLATE
unsigned long long RTREE_QUAL::RemoveAllCount()
{
    // Delete all existing nodes and also count the number of nodes deleted
    unsigned long long count = RemoveAllRecCount(m_root);

    m_root = AllocNode();
    m_root->m_level = 0;
    return count;
}

RTREE_TEMPLATE
void swap(RTREE_QUAL& first, RTREE_QUAL& second)
{
    using std::swap;
    swap(first.m_root, second.m_root);
    swap(first.m_unitSphereVolume, second.m_unitSphereVolume);
}


RTREE_TEMPLATE
void RTREE_QUAL::Reset()
{
#ifdef RTREE_DONT_USE_MEMPOOLS
    // Delete all existing nodes
    RemoveAllRec(m_root);
#else // RTREE_DONT_USE_MEMPOOLS
    // Just reset memory pools.  We are not using complex types
  // EXAMPLE
#endif // RTREE_DONT_USE_MEMPOOLS
}


RTREE_TEMPLATE
void RTREE_QUAL::RemoveAllRec(Node* a_node)
{
    // Allows deallocation of a null root node
    if (a_node == m_root && a_node == NULL) return;

    ASSERT(a_node);
    ASSERT(a_node->m_level >= 0);

    if(a_node->IsInternalNode()) // This is an internal node in the tree
    {
        for(int index=0; index < a_node->m_count; ++index)
        {
            RemoveAllRec(a_node->m_branch[index].m_child);
        }
    }
    FreeNode(a_node);
}

RTREE_TEMPLATE
int RTREE_QUAL::RemoveAllRecCount(Node* a_node)
{
    // Allows deallocation of a null root node
    if (a_node == m_root && a_node == NULL) return 0;
    ASSERT(a_node);
    ASSERT(a_node->m_level >= 0);
    int count = a_node->m_count;
    if(a_node->IsInternalNode()) // This is an internal node in the tree
    {
        for(int index=0; index < a_node->m_count; ++index)
        {
            count += RemoveAllRecCount(a_node->m_branch[index].m_child);
        }
    }
    FreeNode(a_node);
    return count;
}

RTREE_TEMPLATE
typename RTREE_QUAL::Node* RTREE_QUAL::AllocNode()
{
    Node* newNode;
#ifdef RTREE_DONT_USE_MEMPOOLS
    newNode = new Node;
#else // RTREE_DONT_USE_MEMPOOLS
    // EXAMPLE
#endif // RTREE_DONT_USE_MEMPOOLS
    InitNode(newNode);
    return newNode;
}


RTREE_TEMPLATE
void RTREE_QUAL::FreeNode(Node* a_node)
{
            ASSERT(a_node);

#ifdef RTREE_DONT_USE_MEMPOOLS
    //if (a_node->m_count != a_node->getCurrentCount())  {
    //    throw std::invalid_argument("Node m_count do not match the one in the object tracker!");
    //}
    delete a_node;
#else // RTREE_DONT_USE_MEMPOOLS
    // EXAMPLE
#endif // RTREE_DONT_USE_MEMPOOLS
}


// Allocate space for a node in the list used in DeletRect to
// store Nodes that are too empty.
RTREE_TEMPLATE
typename RTREE_QUAL::ListNode* RTREE_QUAL::AllocListNode()
{
#ifdef RTREE_DONT_USE_MEMPOOLS
    return new ListNode;
#else // RTREE_DONT_USE_MEMPOOLS
    // EXAMPLE
#endif // RTREE_DONT_USE_MEMPOOLS
}


RTREE_TEMPLATE
void RTREE_QUAL::FreeListNode(ListNode* a_listNode)
{
#ifdef RTREE_DONT_USE_MEMPOOLS
    delete a_listNode;
#else // RTREE_DONT_USE_MEMPOOLS
    // EXAMPLE
#endif // RTREE_DONT_USE_MEMPOOLS
}


RTREE_TEMPLATE
void RTREE_QUAL::InitNode(Node* a_node)
{
    a_node->m_count = 0;
    a_node->m_level = -1;
}


RTREE_TEMPLATE
void RTREE_QUAL::InitRect(Rect* a_rect)
{
    *a_rect->setEmpty();
}


// Inserts a new data rectangle into the index structure.
// Recursively descends tree, propagates splits back up.
// Returns 0 if node was not split.  Old node updated.
// If node was split, returns 1 and sets the pointer pointed to by
// new_node to point to the new node.  Old node updated to become one of two.
// The level argument specifies the number of steps up from the leaf
// level to insert; e.g. a data rectangle goes in at level = 0.
RTREE_TEMPLATE
bool RTREE_QUAL::InsertRectRec(const Branch& a_branch, Node* a_node, Node** a_newNode, int a_level)
{
            ASSERT(a_node && a_newNode);
            ASSERT(a_level >= 0 && a_level <= a_node->m_level);

    // recurse until we reach the correct level for the new record. data records
    // will always be called with a_level == 0 (leaf)
    if(a_node->m_level > a_level)
    {
        // Still above level for insertion, go down tree recursively
        Node* otherNode;

        // find the optimal branch for this record
        int index = PickBranch(&a_branch.m_rect, a_node);

        // recursively insert this record into the picked branch
        bool childWasSplit = InsertRectRec(a_branch, a_node->m_branch[index].m_child, &otherNode, a_level);

        if (!childWasSplit)
        {
            // Child was not split. Merge the bounding box of the new record with the
            // existing bounding box
            a_node->m_branch[index].m_rect = CombineRect(&a_branch.m_rect, &(a_node->m_branch[index].m_rect));
            return false;
        }
        else
        {
            // Child was split. The old branches are now re-partitioned to two nodes
            // so we have to re-calculate the bounding boxes of each node
            a_node->m_branch[index].m_rect = NodeCover(a_node->m_branch[index].m_child);
            Branch branch;
            branch.m_child = otherNode;
            branch.m_rect = NodeCover(otherNode);

            // The old node is already a child of a_node. Now add the newly-created
            // node to a_node as well. a_node might be split because of that.
            return AddBranch(&branch, a_node, a_newNode);
        }
    }
    else if(a_node->m_level == a_level)
    {
        // We have reached level for insertion. Add rect, split if necessary
        return AddBranch(&a_branch, a_node, a_newNode);
    }
    else
    {
        // Should never occur
                ASSERT(0);
        return false;
    }
}


// Insert a data rectangle into an index structure.
// InsertRect provides for splitting the root;
// returns 1 if root was split, 0 if it was not.
// The level argument specifies the number of steps up from the leaf
// level to insert; e.g. a data rectangle goes in at level = 0.
// InsertRect2 does the recursion.
//
RTREE_TEMPLATE
bool RTREE_QUAL::InsertRect(const Branch& a_branch, Node** a_root, int a_level)
{
    ASSERT(a_root);
    ASSERT(a_level >= 0 && a_level <= (*a_root)->m_level);

    Node* newNode;

    if(InsertRectRec(a_branch, *a_root, &newNode, a_level))  // Root split
    {
        // Grow tree taller and new root
        Node* newRoot = AllocNode();
        newRoot->m_level = (*a_root)->m_level + 1;

        Branch branch;

        // add old root node as a child of the new root
        branch.m_rect = NodeCover(*a_root);
        branch.m_child = *a_root;
        AddBranch(&branch, newRoot, NULL);

        // add the split node as a child of the new root
        branch.m_rect = NodeCover(newNode);
        branch.m_child = newNode;
        AddBranch(&branch, newRoot, NULL);

        // set the new root as the root node
        *a_root = newRoot;

        return true;
    }

    return false;
}


// Find the smallest rectangle that includes all rectangles in branches of a node.
RTREE_TEMPLATE
typename RTREE_QUAL::Rect RTREE_QUAL::NodeCover(Node* a_node)
{
            ASSERT(a_node);

    Rect rect = a_node->m_branch[0].m_rect;
    for(int index = 1; index < a_node->m_count; ++index)
    {
        rect = CombineRect(&rect, &(a_node->m_branch[index].m_rect));
    }

    return rect;
}


// Add a branch to a node.  Split the node if necessary.
// Returns 0 if node not split.  Old node updated.
// Returns 1 if node split, sets *new_node to address of new node.
// Old node updated, becomes one of two.
RTREE_TEMPLATE
bool RTREE_QUAL::AddBranch(const Branch* a_branch, Node* a_node, Node** a_newNode)
{
            ASSERT(a_branch);
            ASSERT(a_node);

    if(a_node->m_count < MAXNODES)  // Split won't be necessary
    {
        a_node->m_branch[a_node->m_count] = *a_branch;
        ++a_node->m_count;
        return false;
    }
    else
    {
                ASSERT(a_newNode);

        SplitNode(a_node, a_branch, a_newNode);
        return true;
    }
}


// Disconnect a dependent node.
// Caller must return (or stop using iteration index) after this as count has changed
RTREE_TEMPLATE
void RTREE_QUAL::DisconnectBranch(Node* a_node, int a_index)
{
            ASSERT(a_node && (a_index >= 0) && (a_index < MAXNODES));
            ASSERT(a_node->m_count > 0);

    // Remove element by swapping with the last element to prevent gaps in array
    a_node->m_branch[a_index] = a_node->m_branch[a_node->m_count - 1];

    --a_node->m_count;
}


// Pick a branch.  Pick the one that will need the smallest increase
// in area to accomodate the new rectangle.  This will result in the
// least total area for the covering rectangles in the current node.
// In case of a tie, pick the one which was smaller before, to get
// the best resolution when searching.
RTREE_TEMPLATE
int RTREE_QUAL::PickBranch(const Rect* a_rect, Node* a_node)
{
            ASSERT(a_rect && a_node);

    bool firstTime = true;
    ELEMTYPEREAL increase;
    ELEMTYPEREAL bestIncr = (ELEMTYPEREAL)-1;
    ELEMTYPEREAL area;
    ELEMTYPEREAL bestArea;
    int best = 0;
    Rect tempRect;

    for(int index=0; index < a_node->m_count; ++index)
    {
        Rect* curRect = &a_node->m_branch[index].m_rect;
        area = CalcRectVolume(curRect);
        tempRect = CombineRect(a_rect, curRect);
        increase = CalcRectVolume(&tempRect) - area;
        if((increase < bestIncr) || firstTime)
        {
            best = index;
            bestArea = area;
            bestIncr = increase;
            firstTime = false;
        }
        else if((increase == bestIncr) && (area < bestArea))
        {
            best = index;
            bestArea = area;
            bestIncr = increase;
        }
    }
    return best;
}


// Combine two rectangles into larger one containing both
RTREE_TEMPLATE
typename RTREE_QUAL::Rect RTREE_QUAL::CombineRect(const Rect* a_rectA, const Rect* a_rectB)
{
    ASSERT(a_rectA && a_rectB);
    Rect newRect = *a_rectA;
    return newRect.extend(*a_rectB);
}


// Split a node.
// Divides the nodes branches and the extra one between two nodes.
// Old node is one of the new ones, and one really new one is created.
// Tries more than one method for choosing a partition, uses best result.
RTREE_TEMPLATE
void RTREE_QUAL::SplitNode(Node* a_node, const Branch* a_branch, Node** a_newNode)
{
            ASSERT(a_node);
            ASSERT(a_branch);

    // Could just use local here, but member or external is faster since it is reused
    PartitionVars localVars;
    PartitionVars* parVars = &localVars;

    // Load all the branches into a buffer, initialize old node
    GetBranches(a_node, a_branch, parVars);

    // Find partition
    ChoosePartition(parVars, MINNODES);

    // Create a new node to hold (about) half of the branches
    *a_newNode = AllocNode();
    (*a_newNode)->m_level = a_node->m_level;

    // Put branches from buffer into 2 nodes according to the chosen partition
    a_node->m_count = 0;
    LoadNodes(a_node, *a_newNode, parVars);

            ASSERT((a_node->m_count + (*a_newNode)->m_count) == parVars->m_total);
}


// Calculate the n-dimensional volume of a rectangle
RTREE_TEMPLATE
ELEMTYPEREAL RTREE_QUAL::RectVolume(Rect* a_rect)
{
    ASSERT(a_rect);

    ELEMTYPEREAL volume = a_rect->volume();

    ASSERT(volume >= (ELEMTYPEREAL)0);

    return volume;
}


// The exact volume of the bounding sphere for the given Rect
RTREE_TEMPLATE
ELEMTYPEREAL RTREE_QUAL::RectSphericalVolume(Rect* a_rect)
{
    ASSERT(a_rect);

    Rect_Real_Vec Extent = 0.5 * (a_rect->max() - a_rect->min()).template cast <ELEMTYPEREAL>();
    ELEMTYPEREAL sumOfSquares = Extent.dot(Extent);
    ELEMTYPEREAL radius = (ELEMTYPEREAL) sqrt(sumOfSquares);

    // Pow maybe slow, so test for common dims like 2,3 and just use x*x, x*x*x.
    if(NUMDIMS == 3)
    {
        return (radius * radius * radius * m_unitSphereVolume);
    }
    else if(NUMDIMS == 2)
    {
        return (radius * radius * m_unitSphereVolume);
    }
    else
    {
        return (ELEMTYPEREAL)(pow(radius, NUMDIMS) * m_unitSphereVolume);
    }
}


// Use one of the methods to calculate retangle volume
RTREE_TEMPLATE
ELEMTYPEREAL RTREE_QUAL::CalcRectVolume(Rect* a_rect)
{
#ifdef RTREE_USE_SPHERICAL_VOLUME
    return RectSphericalVolume(a_rect); // Slower but helps certain merge cases
#else // RTREE_USE_SPHERICAL_VOLUME
    return RectVolume(a_rect); // Faster but can cause poor merges
#endif // RTREE_USE_SPHERICAL_VOLUME
}


// Load branch buffer with branches from full node plus the extra branch.
RTREE_TEMPLATE
void RTREE_QUAL::GetBranches(Node* a_node, const Branch* a_branch, PartitionVars* a_parVars)
{
            ASSERT(a_node);
            ASSERT(a_branch);

            ASSERT(a_node->m_count == MAXNODES);

    // Load the branch buffer
    for(int index=0; index < MAXNODES; ++index)
    {
        a_parVars->m_branchBuf[index] = a_node->m_branch[index];
    }
    a_parVars->m_branchBuf[MAXNODES] = *a_branch;
    a_parVars->m_branchCount = MAXNODES + 1;

    // Calculate rect containing all in the set
    a_parVars->m_coverSplit = a_parVars->m_branchBuf[0].m_rect;
    for(int index=1; index < MAXNODES+1; ++index)
    {
        a_parVars->m_coverSplit = CombineRect(&a_parVars->m_coverSplit, &a_parVars->m_branchBuf[index].m_rect);
    }
    a_parVars->m_coverSplitArea = CalcRectVolume(&a_parVars->m_coverSplit);
}


// Method #0 for choosing a partition:
// As the seeds for the two groups, pick the two rects that would waste the
// most area if covered by a single rectangle, i.e. evidently the worst pair
// to have in the same group.
// Of the remaining, one at a time is chosen to be put in one of the two groups.
// The one chosen is the one with the greatest difference in area expansion
// depending on which group - the rect most strongly attracted to one group
// and repelled from the other.
// If one group gets too full (more would force other group to violate min
// fill requirement) then other group gets the rest.
// These last are the ones that can go in either group most easily.
RTREE_TEMPLATE
void RTREE_QUAL::ChoosePartition(PartitionVars* a_parVars, int a_minFill)
{
            ASSERT(a_parVars);

    ELEMTYPEREAL biggestDiff;
    int group, chosen = 0, betterGroup = 0;

    InitParVars(a_parVars, a_parVars->m_branchCount, a_minFill);
    PickSeeds(a_parVars);

    while (((a_parVars->m_count[0] + a_parVars->m_count[1]) < a_parVars->m_total)
           && (a_parVars->m_count[0] < (a_parVars->m_total - a_parVars->m_minFill))
           && (a_parVars->m_count[1] < (a_parVars->m_total - a_parVars->m_minFill)))
    {
        biggestDiff = (ELEMTYPEREAL) -1;
        for(int index=0; index<a_parVars->m_total; ++index)
        {
            if(PartitionVars::NOT_TAKEN == a_parVars->m_partition[index])
            {
                Rect* curRect = &a_parVars->m_branchBuf[index].m_rect;
                Rect rect0 = CombineRect(curRect, &a_parVars->m_cover[0]);
                Rect rect1 = CombineRect(curRect, &a_parVars->m_cover[1]);
                ELEMTYPEREAL growth0 = CalcRectVolume(&rect0) - a_parVars->m_area[0];
                ELEMTYPEREAL growth1 = CalcRectVolume(&rect1) - a_parVars->m_area[1];
                ELEMTYPEREAL diff = growth1 - growth0;
                if(diff >= 0)
                {
                    group = 0;
                }
                else
                {
                    group = 1;
                    diff = -diff;
                }

                if(diff > biggestDiff)
                {
                    biggestDiff = diff;
                    chosen = index;
                    betterGroup = group;
                }
                else if((diff == biggestDiff) && (a_parVars->m_count[group] < a_parVars->m_count[betterGroup]))
                {
                    chosen = index;
                    betterGroup = group;
                }
            }
        }
        Classify(chosen, betterGroup, a_parVars);
    }

    // If one group too full, put remaining rects in the other
    if((a_parVars->m_count[0] + a_parVars->m_count[1]) < a_parVars->m_total)
    {
        if(a_parVars->m_count[0] >= a_parVars->m_total - a_parVars->m_minFill)
        {
            group = 1;
        }
        else
        {
            group = 0;
        }
        for(int index=0; index<a_parVars->m_total; ++index)
        {
            if(PartitionVars::NOT_TAKEN == a_parVars->m_partition[index])
            {
                Classify(index, group, a_parVars);
            }
        }
    }

            ASSERT((a_parVars->m_count[0] + a_parVars->m_count[1]) == a_parVars->m_total);
            ASSERT((a_parVars->m_count[0] >= a_parVars->m_minFill) &&
                   (a_parVars->m_count[1] >= a_parVars->m_minFill));
}


// Copy branches from the buffer into two nodes according to the partition.
RTREE_TEMPLATE
void RTREE_QUAL::LoadNodes(Node* a_nodeA, Node* a_nodeB, PartitionVars* a_parVars)
{
            ASSERT(a_nodeA);
            ASSERT(a_nodeB);
            ASSERT(a_parVars);

    for(int index=0; index < a_parVars->m_total; ++index)
    {
                ASSERT(a_parVars->m_partition[index] == 0 || a_parVars->m_partition[index] == 1);

        int targetNodeIndex = a_parVars->m_partition[index];
        Node* targetNodes[] = {a_nodeA, a_nodeB};

        // It is assured that AddBranch here will not cause a node split.
        bool nodeWasSplit = AddBranch(&a_parVars->m_branchBuf[index], targetNodes[targetNodeIndex], NULL);
                ASSERT(!nodeWasSplit);
    }
}


// Initialize a PartitionVars structure.
RTREE_TEMPLATE
void RTREE_QUAL::InitParVars(PartitionVars* a_parVars, int a_maxRects, int a_minFill)
{
            ASSERT(a_parVars);

    a_parVars->m_count[0] = a_parVars->m_count[1] = 0;
    a_parVars->m_area[0] = a_parVars->m_area[1] = (ELEMTYPEREAL)0;
    a_parVars->m_total = a_maxRects;
    a_parVars->m_minFill = a_minFill;
    for(int index=0; index < a_maxRects; ++index)
    {
        a_parVars->m_partition[index] = PartitionVars::NOT_TAKEN;
    }
}


RTREE_TEMPLATE
void RTREE_QUAL::PickSeeds(PartitionVars* a_parVars)
{
    int seed0 = 0, seed1 = 0;
    ELEMTYPEREAL worst, waste;
    ELEMTYPEREAL area[MAXNODES+1];

    for(int index=0; index<a_parVars->m_total; ++index)
    {
        area[index] = CalcRectVolume(&a_parVars->m_branchBuf[index].m_rect);
    }

    worst = -a_parVars->m_coverSplitArea - 1;
    for(int indexA=0; indexA < a_parVars->m_total-1; ++indexA)
    {
        for(int indexB = indexA+1; indexB < a_parVars->m_total; ++indexB)
        {
            Rect oneRect = CombineRect(&a_parVars->m_branchBuf[indexA].m_rect, &a_parVars->m_branchBuf[indexB].m_rect);
            waste = CalcRectVolume(&oneRect) - area[indexA] - area[indexB];
            if(waste > worst)
            {
                worst = waste;
                seed0 = indexA;
                seed1 = indexB;
            }
        }
    }

    Classify(seed0, 0, a_parVars);
    Classify(seed1, 1, a_parVars);
}


// Put a branch in one of the groups.
RTREE_TEMPLATE
void RTREE_QUAL::Classify(int a_index, int a_group, PartitionVars* a_parVars)
{
            ASSERT(a_parVars);
            ASSERT(PartitionVars::NOT_TAKEN == a_parVars->m_partition[a_index]);

    a_parVars->m_partition[a_index] = a_group;

    // Calculate combined rect
    if (a_parVars->m_count[a_group] == 0)
    {
        a_parVars->m_cover[a_group] = a_parVars->m_branchBuf[a_index].m_rect;
    }
    else
    {
        a_parVars->m_cover[a_group] = CombineRect(&a_parVars->m_branchBuf[a_index].m_rect, &a_parVars->m_cover[a_group]);
    }

    // Calculate volume of combined rect
    a_parVars->m_area[a_group] = CalcRectVolume(&a_parVars->m_cover[a_group]);

    ++a_parVars->m_count[a_group];
}


// Delete a data rectangle from an index structure.
// Pass in a pointer to a Rect, the tid of the record, ptr to ptr to root node.
// Returns 1 if record not found, 0 if success.
// RemoveRect provides for eliminating the root.
RTREE_TEMPLATE
bool RTREE_QUAL::RemoveRect(Rect const *a_rect, const DATATYPE& a_id, Node** a_root)
{
            ASSERT(a_rect && a_root);
            ASSERT(*a_root);

    ListNode* reInsertList = NULL;

    if(!RemoveRectRec(a_rect, a_id, *a_root, &reInsertList))
    {
        // Found and deleted a data item
        // Reinsert any branches from eliminated nodes
        while(reInsertList)
        {
            Node* tempNode = reInsertList->m_node;

            for(int index = 0; index < tempNode->m_count; ++index)
            {
                // TODO go over this code. should I use (tempNode->m_level - 1)?
                InsertRect(tempNode->m_branch[index],
                           a_root,
                           tempNode->m_level);
            }

            ListNode* remLNode = reInsertList;
            reInsertList = reInsertList->m_next;

            FreeNode(remLNode->m_node);
            FreeListNode(remLNode);
        }

        // Check for redundant root (not leaf, 1 child) and eliminate TODO replace
        // if with while? In case there is a whole branch of redundant roots...
        if((*a_root)->m_count == 1 && (*a_root)->IsInternalNode())
        {
            Node* tempNode = (*a_root)->m_branch[0].m_child;

                    ASSERT(tempNode);
            FreeNode(*a_root);
            *a_root = tempNode;
        }
        return false;
    }
    else
    {
        return true;
    }
}


// Delete a rectangle from non-root part of an index structure.
// Called by RemoveRect.  Descends tree recursively,
// merges branches on the way back up.
// Returns 1 if record not found, 0 if success.
RTREE_TEMPLATE
bool RTREE_QUAL::RemoveRectRec(Rect const *a_rect, const DATATYPE& a_id, Node* a_node, ListNode** a_listNode)
{
            ASSERT(a_rect && a_node && a_listNode);
            ASSERT(a_node->m_level >= 0);

    // If a_id is found, return false
    // As long as a_rect intersects the bounding box of a_id, a_id will be found!

    if(a_node->IsInternalNode())  // not a leaf node
    {
        for(int index = 0; index < a_node->m_count; ++index)
        {
            if(Overlap(a_rect, &(a_node->m_branch[index].m_rect)))
            {
                if(!RemoveRectRec(a_rect, a_id, a_node->m_branch[index].m_child, a_listNode))
                {
                    if(a_node->m_branch[index].m_child->m_count >= MINNODES)
                    {
                        // child removed, just resize parent rect
                        a_node->m_branch[index].m_rect = NodeCover(a_node->m_branch[index].m_child);
                    }
                    else
                    {
                        // child removed, not enough entries in node, eliminate node
                        ReInsert(a_node->m_branch[index].m_child, a_listNode);
                        DisconnectBranch(a_node, index); // Must return after this call as count has changed
                    }
                    return false;
                }
            }
        }
        return true;
    }
    else // A leaf node
    {
        for(int index = 0; index < a_node->m_count; ++index)
        {
            if(a_node->m_branch[index].m_data == a_id)
            {
                DisconnectBranch(a_node, index); // Must return after this call as count has changed
                return false;
            }
        }
        return true;
    }
}


// Decide whether two rectangles overlap.
RTREE_TEMPLATE
bool RTREE_QUAL::Overlap(Rect const* a_rectA, Rect const* a_rectB) const
{
    ASSERT(a_rectA && a_rectB);
    return a_rectA->intersects(*a_rectB);
}


// Add a node to the reinsertion list.  All its branches will later
// be reinserted into the index structure.
RTREE_TEMPLATE
void RTREE_QUAL::ReInsert(Node* a_node, ListNode** a_listNode)
{
    ListNode* newListNode;

    newListNode = AllocListNode();
    newListNode->m_node = a_node;
    newListNode->m_next = *a_listNode;
    *a_listNode = newListNode;
}

RTREE_TEMPLATE
int RTREE_QUAL::Search(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], t_resultCallback a_resultCallback, void* a_context)
{
    Rect rect(a_min, a_max);

    // NOTE: May want to return search result another way, perhaps returning the number of found elements here.

    int foundCount = 0;
    Search(m_root, &rect, foundCount, a_resultCallback, a_context);

    return foundCount;
}

// Search in an index tree or subtree for all data retangles that overlap the argument rectangle.
RTREE_TEMPLATE
bool RTREE_QUAL::Search(Node* a_node, Rect const*a_rect, int& a_foundCount, std::function<bool (const DATATYPE&)> callback) const
{
            ASSERT(a_node);
            ASSERT(a_node->m_level >= 0);
            ASSERT(a_rect);

    if(a_node->IsInternalNode())
    {
        // This is an internal node in the tree
        for(int index=0; index < a_node->m_count; ++index)
        {
            if(Overlap(a_rect, &a_node->m_branch[index].m_rect))
            {
                if(!Search(a_node->m_branch[index].m_child, a_rect, a_foundCount, callback))
                {
                    // The callback indicated to stop searching
                    return false;
                }
            }
        }
    }
    else
    {
        // This is a leaf node
        for(int index=0; index < a_node->m_count; ++index)
        {
            if(Overlap(a_rect, &a_node->m_branch[index].m_rect))
            {
                DATATYPE& id = a_node->m_branch[index].m_data;
                ++a_foundCount;

                if(callback && !callback(id))
                {
                    return false; // Don't continue searching
                }
            }
        }
    }

    return true; // Continue searching
}

// Search in an index tree or subtree for all data retangles that overlap the argument rectangle.
RTREE_TEMPLATE
bool RTREE_QUAL::Search(Node* a_node, Rect* a_rect, int& a_foundCount, t_resultCallback a_resultCallback, void* a_context)
{
            ASSERT(a_node);
            ASSERT(a_node->m_level >= 0);
            ASSERT(a_rect);

    if(a_node->IsInternalNode())
    {
        // This is an internal node in the tree
        for(int index=0; index < a_node->m_count; ++index)
        {
            if(Overlap(a_rect, &a_node->m_branch[index].m_rect))
            {
                if(!Search(a_node->m_branch[index].m_child, a_rect, a_foundCount, a_resultCallback, a_context))
                {
                    // The callback indicated to stop searching
                    return false;
                }
            }
        }
    }
    else
    {
        // This is a leaf node
        for(int index=0; index < a_node->m_count; ++index)
        {
            if(Overlap(a_rect, &a_node->m_branch[index].m_rect))
            {
                DATATYPE& id = a_node->m_branch[index].m_data;
                ++a_foundCount;

                // NOTE: There are different ways to return results.  Here's where to modify
                if(a_resultCallback)
                {
                    if(!a_resultCallback(id, a_context))
                    {
                        return false; // Don't continue searching
                    }
                }
            }
        }
    }

    return true; // Continue searching
}

// Search in an index tree or subtree for all data retangles that overlap the argument rectangle.
RTREE_TEMPLATE
bool RTREE_QUAL::Search(Node* a_node, Rect const *a_rect, std::vector<DATATYPE>& results) const
{
            ASSERT(a_node);
            ASSERT(a_node->m_level >= 0);
            ASSERT(a_rect);

    if(a_node->IsInternalNode())
    {
        // This is an internal node in the tree
        for(int index=0; index < a_node->m_count; ++index)
        {
            if(Overlap(a_rect, &a_node->m_branch[index].m_rect))
            {
                Search(a_node->m_branch[index].m_child, a_rect, results);
            }
        }
    }
    else
    {
        // This is a leaf node
        for(int index=0; index < a_node->m_count; ++index)
        {
            if(Overlap(a_rect, &a_node->m_branch[index].m_rect))
            {
                results.push_back(a_node->m_branch[index].m_data);
            }
        }
    }
    return true; // Continue searching
}

RTREE_TEMPLATE
void RTREE_QUAL::CopyRec(Node* current, Node* other)
{
    current->m_level = other->m_level;
    current->m_count = other->m_count;
    if(current->IsInternalNode())  // not a leaf node
    {
        for(int index = 0; index < current->m_count; ++index)
        {
            Branch* currentBranch = &current->m_branch[index];
            Branch* otherBranch = &other->m_branch[index];

            currentBranch->m_rect = otherBranch->m_rect;

            currentBranch->m_child = AllocNode();
            CopyRec(currentBranch->m_child, otherBranch->m_child);
        }
    }
    else // A leaf node
    {
        for(int index = 0; index < current->m_count; ++index)
        {
            Branch* currentBranch = &current->m_branch[index];
            Branch* otherBranch = &other->m_branch[index];

            currentBranch->m_rect = otherBranch->m_rect;

            currentBranch->m_data = otherBranch->m_data;
        }
    }
}

