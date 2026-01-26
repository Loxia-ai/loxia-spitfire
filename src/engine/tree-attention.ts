/**
 * Tree-Parallel Speculative Decoding
 *
 * Process an entire speculation tree in ONE forward pass by using
 * a tree-structured attention mask where each node attends only to its ancestors.
 */

export interface TreeNode {
  tokenId: number;
  parentIndex: number;  // -1 for root's children
  depth: number;
  index: number;        // Position in flattened tree
}

export interface SpeculationTree {
  nodes: TreeNode[];
  depth: number;
  branchingFactor: number;
}

/**
 * Build a speculation tree from top-k candidates at each level
 *
 * @param rootCandidates - Top-k tokens for first level (from current logits)
 * @param depth - How many levels deep to speculate
 * @param branchingFactor - How many candidates per node (top-k)
 * @param getCandidates - Function to get top-k candidates given a token
 * @returns Flattened tree structure
 */
export function buildSpeculationTree(
  rootCandidates: number[],
  depth: number,
  branchingFactor: number,
  getCandidates: (tokenId: number) => number[]
): SpeculationTree {
  const nodes: TreeNode[] = [];

  // Level 1: direct children of current position
  for (let i = 0; i < rootCandidates.length; i++) {
    nodes.push({
      tokenId: rootCandidates[i],
      parentIndex: -1,  // Root's children
      depth: 1,
      index: nodes.length,
    });
  }

  // Build subsequent levels
  let levelStart = 0;
  let levelEnd = nodes.length;

  for (let d = 2; d <= depth; d++) {
    for (let i = levelStart; i < levelEnd; i++) {
      const parent = nodes[i];
      const children = getCandidates(parent.tokenId);

      for (let j = 0; j < Math.min(children.length, branchingFactor); j++) {
        nodes.push({
          tokenId: children[j],
          parentIndex: i,
          depth: d,
          index: nodes.length,
        });
      }
    }

    levelStart = levelEnd;
    levelEnd = nodes.length;
  }

  return { nodes, depth, branchingFactor };
}

/**
 * Generate tree attention mask
 * mask[i][j] = 1 if node i can attend to node j, 0 otherwise
 * A node can attend to itself and all its ancestors
 *
 * @param tree - The speculation tree
 * @param seqLenBefore - Positions before the tree (always attendable)
 * @returns Flattened attention mask
 */
export function generateTreeAttentionMask(
  tree: SpeculationTree,
  seqLenBefore: number
): Float32Array {
  const treeSize = tree.nodes.length;
  const totalSize = seqLenBefore + treeSize;

  // Mask is [treeSize, totalSize] - for each tree node, which positions can it attend to
  const mask = new Float32Array(treeSize * totalSize);

  for (let i = 0; i < treeSize; i++) {
    const node = tree.nodes[i];
    const rowStart = i * totalSize;

    // Can attend to all positions before the tree (the committed prefix)
    for (let j = 0; j < seqLenBefore; j++) {
      mask[rowStart + j] = 1.0;
    }

    // Can attend to self
    mask[rowStart + seqLenBefore + i] = 1.0;

    // Can attend to ancestors
    let ancestorIdx = node.parentIndex;
    while (ancestorIdx >= 0) {
      mask[rowStart + seqLenBefore + ancestorIdx] = 1.0;
      ancestorIdx = tree.nodes[ancestorIdx].parentIndex;
    }
  }

  return mask;
}

/**
 * Find the longest valid path in the tree given verification results
 *
 * @param tree - The speculation tree
 * @param actualTokens - What the model actually predicted at each position
 * @returns Array of accepted token IDs along the longest valid path
 */
export function findLongestValidPath(
  tree: SpeculationTree,
  actualTokens: number[]  // actualTokens[i] = what model predicted after processing node i
): number[] {
  // For each node, check if it matches what the parent's position predicted
  const validNodes = new Set<number>();

  // First level nodes are valid if they match what the root position predicted
  // (we don't have that info here - caller should handle first level separately)
  // For now, assume all first-level nodes are "valid" and check subsequent levels

  for (let i = 0; i < tree.nodes.length; i++) {
    const node = tree.nodes[i];

    if (node.parentIndex === -1) {
      // First level - assume valid (caller verifies against previous logits)
      validNodes.add(i);
    } else {
      // Check if this node's token matches what parent predicted
      const parentPrediction = actualTokens[node.parentIndex];
      if (parentPrediction === node.tokenId && validNodes.has(node.parentIndex)) {
        validNodes.add(i);
      }
    }
  }

  // Find the deepest valid node
  let bestNode: TreeNode | null = null;
  for (const idx of validNodes) {
    const node = tree.nodes[idx];
    if (!bestNode || node.depth > bestNode.depth) {
      bestNode = node;
    }
  }

  if (!bestNode) {
    return [];
  }

  // Trace back to root to get the path
  const path: number[] = [];
  let current: TreeNode | null = bestNode;
  while (current) {
    path.unshift(current.tokenId);
    current = current.parentIndex >= 0 ? tree.nodes[current.parentIndex] : null;
  }

  return path;
}

/**
 * Get the token IDs from the tree in order (for forward pass)
 */
export function getTreeTokenIds(tree: SpeculationTree): number[] {
  return tree.nodes.map(n => n.tokenId);
}

/**
 * Calculate tree statistics
 */
export function getTreeStats(tree: SpeculationTree): {
  totalNodes: number;
  nodesPerLevel: number[];
} {
  const nodesPerLevel: number[] = [];
  for (const node of tree.nodes) {
    while (nodesPerLevel.length <= node.depth) {
      nodesPerLevel.push(0);
    }
    nodesPerLevel[node.depth]++;
  }

  return {
    totalNodes: tree.nodes.length,
    nodesPerLevel,
  };
}
