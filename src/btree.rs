use std::collections::VecDeque;
use std::fmt::Debug;
use std::cmp::{Ord, Ordering};
use thunderdome::{Arena, Index as ArenaNodeIndex};

#[derive(Debug)]
pub struct BPlusTree<K, V> {
    arena:                      Arena<BPlusTreeNode<K, V>>,
    root:                       Option<ArenaNodeIndex>,
    // min_keys = b - 1 && max_keys = 2 * b - 1. b > 1
    b:                          usize,
    balance_siblings_per_side:  usize
}

// to reduce confusion between indextree's types this is made more explicit
#[derive(Debug, Clone)]
struct BPlusTreeNode<K, V> {
    keys:       Vec<K>,
    // for internal nodes values will be not present and the vec will not allocate any space on heap by default
    // https://doc.rust-lang.org/std/vec/struct.Vec.html#guarantees
    values:     Vec<V>,
    // for leaf nodes children will not be present
    children:   Vec<ArenaNodeIndex>,
    left:       Option<ArenaNodeIndex>,
    right:      Option<ArenaNodeIndex>,
}

#[derive(Debug)]
struct PathDetail {
    parent_id:          ArenaNodeIndex,
    child_id:           ArenaNodeIndex,     // this is here alongside the child_index to avoid vec access
    child_index:        usize,              // this is the index in `BPlusTreeNode.children` array.
}

enum SearchResult {
    Found(usize),
    GoDown(usize)
}

#[derive(Debug)]
struct Sibling<K, V> {
    id:     ArenaNodeIndex,
    node:   BPlusTreeNode<K, V>
}

impl<K: Ord, V> BPlusTreeNode<K, V> {
    pub fn new() -> Self {
        Self {
            keys: Default::default(), values: Default::default(), children: Default::default(), left: None, right: None
        }
    }

    pub fn search(&self, key: &K) -> SearchResult {
        // For smaller keys linear search will perform *singnificantly* better
        // self.search_linear(key)
        self.search_binary(key)
    }

    fn search_linear(&self, key: &K) -> SearchResult {
        for (i, k) in self.keys.iter().enumerate() {
            match k.cmp(key) {
                Ordering::Equal => return SearchResult::Found(i),
                Ordering::Greater => return SearchResult::GoDown(i),
                Ordering::Less => {},
            }
        }
        SearchResult::GoDown(self.keys.len())
    }

    fn search_binary(&self, key: &K) -> SearchResult {
        let mut low: i32 = 0;
        let mut high: i32 = (self.keys.len() as i32) - 1;
        let mut mid: i32 = 0;

        while low <= high {
            mid = (low + high) / 2;

            match (self.keys[mid as usize]).cmp(key) {
                Ordering::Equal => return SearchResult::Found(mid as usize),
                Ordering::Greater => {
                    high = mid - 1;
                },
                Ordering::Less => {
                    low = mid + 1;
                },
            }
        }

        return SearchResult::GoDown(low as usize);
    }
}

impl<K, V> BPlusTree<K, V>
where K: Ord + Debug + Clone, V: Debug + Clone {
    pub fn new(b: usize) -> Self {
        Self {
            arena: Arena::new(),
            root: None,
            b,
            balance_siblings_per_side: 2
        }
    }

    // search the key until we reach the leaf node and return the search path
    fn search(&self, key: &K) -> Vec<PathDetail> {
        let mut path: Vec<PathDetail> = vec![];
        let mut node_id = self.root.unwrap();
        let mut node = self.arena.get(node_id).unwrap();

        // while we don't reach to leaf node
        while !node.children.is_empty() {
            let search_result = node.search(key);
            let child_index = match search_result {
                SearchResult::GoDown(child_index) => child_index,
                SearchResult::Found(child_index) => child_index,
            };
            let path_detail= PathDetail {parent_id: node_id, child_index, child_id: *node.children.get(child_index).unwrap()};
            node_id = path_detail.child_id;
            node = self.arena.get(node_id).unwrap();
            path.push(path_detail);
        }

        // return the leaf node id
        path
    }

    // get immutable reference to the value
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.root.is_none() {
            return None;
        }

        let mut path = self.search(key);
        let mut leaf_id = self.root.unwrap();
        if let Some(leaf_parent) = path.pop() {
            leaf_id         = leaf_parent.child_id;
        }

        let leaf_node = self.arena.get(leaf_id).unwrap();

        match leaf_node.search(&key) {
            SearchResult::Found(key_index) => leaf_node.values.get(key_index),
            _ => None
        }
    }

    // @todo we should be able to query when start or end both are not present
    pub fn get_in_range(&self, start: &K, end: &K) -> Vec<&V> {
        let mut values = Vec::<&V>::new();
        if self.root.is_none() {
            return values;
        }

        let mut path = self.search(start);
        let mut leaf_id = self.root.unwrap();
        if let Some(leaf_parent) = path.pop() {
            leaf_id         = leaf_parent.child_id;
        }

        let mut leaf_node = self.arena.get(leaf_id).unwrap();
        match leaf_node.search(start) {
            SearchResult::Found(key_index) => {
                // first copy values of current leaf node from key_index
                for (key, value) in leaf_node.keys.iter().skip(key_index).zip(leaf_node.values.iter().skip(key_index)) {
                    if *key <= *end {
                        values.push(value);
                    } else {
                        return values;
                    }
                }

                let mut next_leaf_node_id = leaf_node.right;
                // add values from current leaf node
                while let Some(leaf_id) = next_leaf_node_id {
                    leaf_node = self.arena.get(leaf_id).unwrap();
                    for (key, value) in leaf_node.keys.iter().zip(leaf_node.values.iter()) {
                        if *key <= *end {
                            values.push(value);
                        } else {
                            return values;
                        }
                    }
                    next_leaf_node_id = leaf_node.right;
                }
            },
            _ => {}
        }

        return values;
    }

    // insert a key value pair. If the map did not have this key present, None is returned.
    pub fn insert(&mut self, key: K, value: V) -> Option<V>{
        // if root is empty then create the root node
        if self.root.is_none() {
            self.root = Some(self.arena.insert(BPlusTreeNode::new()));
        }

        let mut path = self.search(&key);
        let mut leaf_id = self.root.unwrap();
        let mut leaf_index = None;
        let mut leaf_parent_id = None;
        if let Some(leaf_parent) = path.pop() {
            leaf_parent_id  = Some(leaf_parent.parent_id);
            leaf_index      = Some(leaf_parent.child_index);
            leaf_id         = leaf_parent.child_id;
        }

        // insert the new key/value inside the leaf node, possibly causing overflow
        let leaf_node = self.arena.get_mut(leaf_id).unwrap();
        match leaf_node.search(&key) {
            SearchResult::Found(insert_at) => {
                //@note: when we will be having variable sized values we need to call balance
                let old_val = Some(leaf_node.values.get(insert_at).unwrap().clone());
                *leaf_node.values.get_mut(insert_at).unwrap() = value;
                old_val
            },
            SearchResult::GoDown(insert_at) => {
                leaf_node.keys.insert(insert_at, key);
                leaf_node.values.insert(insert_at, value);
                // balance
                self.balance(leaf_parent_id, leaf_id, leaf_index, path);
                None
            }
        }
    }

    // Removes a key from the map, returning the value at the key if the key was previously in the map.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if self.root.is_none() {
            return None;
        }

        let mut path = self.search(&key);
        let mut leaf_id = self.root.unwrap();
        let mut leaf_index = None;
        let mut leaf_parent_id = None;
        if let Some(leaf_parent) = path.pop() {
            leaf_parent_id  = Some(leaf_parent.parent_id);
            leaf_index      = Some(leaf_parent.child_index);
            leaf_id         = leaf_parent.child_id;
        }

        let leaf_node = self.arena.get_mut(leaf_id).unwrap();
        match leaf_node.search(&key) {
            SearchResult::Found(key_index) => {
                leaf_node.keys.remove(key_index);
                let value = Some(leaf_node.values.remove(key_index));
                // balance
                self.balance(leaf_parent_id, leaf_id, leaf_index, path);
                value
            },
            _ => None
        }
    }

    fn get_siblings(&mut self, parent: &BPlusTreeNode<K, V>, child_index: usize) -> (Vec<Sibling<K, V>>, usize) {
        let num_siblings_per_side = if child_index == 0 || child_index == parent.children.len() - 1 {
            self.balance_siblings_per_side * 2
        } else {
            self.balance_siblings_per_side
        };

        // Calculate ranges for left and right siblings
        let left_siblings = child_index.saturating_sub(1)..child_index;
        let right_siblings = (child_index + 1)..(child_index + num_siblings_per_side + 1).min(parent.children.len());

        // Pre-allocate the vector for siblings
        let mut siblings = Vec::with_capacity(left_siblings.len() + right_siblings.len() + 1);

        let mut first_sibling_index = None;

        // Collect the siblings: left, then current child, then right siblings
        for index in left_siblings.chain(Some(child_index)).chain(right_siblings) {
            if first_sibling_index.is_none() { first_sibling_index = Some(index); }
            let id = *parent.children.get(index).unwrap();
            let node = self.arena.get(id).unwrap().clone();
            siblings.push(Sibling { id, node });
        }

        (siblings, first_sibling_index.unwrap())
    }

    fn balance(&mut self, parent_id: Option<ArenaNodeIndex>, child_id: ArenaNodeIndex, child_index: Option<usize>, mut path: Vec<PathDetail>) {
        let child           = self.arena.get(child_id).unwrap();
        let child_is_leaf   = child.children.is_empty();
        let is_root         = parent_id.is_none();
        // root is only considered underflow if it has zero keys
        let is_underflow    = if is_root { child.keys.is_empty() } else { child.keys.len() < self.b - 1 };
        let is_overflow     = child.keys.len() > self.b * 2 - 1;

        // check if we need to do anything at all
        if !is_underflow && !is_overflow { return; }

        // root
        if is_root {
            let mut root = self.arena.get(child_id).unwrap().clone();

            if is_underflow {
                // here either root is leaf and hence completely empty OR
                // it can be internal and have only one child
                let mut new_root = None;
                if root.children.len() == 1 {
                    // update root
                    new_root = root.children.pop();
                }
                // free current root
                self.arena.remove(child_id);
                self.root = new_root;
            } else {
                //
                // if current root is a leaf node then we need to keep the pivot both in parent and in left child whereas if it's
                // internal then we need to remove pivot from left so that we are having correct number of children in left
                //

                let is_leaf = root.children.is_empty();
                // we don't move the root node, insted create new left and right
                let mut left = BPlusTreeNode::<K, V>::new();
                let mut right = BPlusTreeNode::<K, V>::new();

                if is_leaf {
                    let keys_per_node: usize = root.keys.len() / 2;
                    // we are having left heavy distribution hence we will add extra keys in left node if needed
                    let extra_keys_in_left_node = root.keys.len() % 2;

                    // distribute
                    right.keys = root.keys.split_off(keys_per_node + extra_keys_in_left_node);
                    right.values = root.values.split_off(keys_per_node + extra_keys_in_left_node);
                    left.keys = root.keys.split_off(0);
                    left.values = root.values.split_off(0);
                    root.keys.push(left.keys.last().unwrap().clone());
                } else {
                    let pivot_index: usize = root.keys.len() / 2; // left-biased

                    // distribute
                    right.keys = root.keys.split_off(pivot_index + 1);
                    // add remaining inside the left
                    left.keys = root.keys.split_off(0);
                    // now add a pivot key inside the root
                    root.keys = vec![left.keys.pop().unwrap()];
                    right.children = root.children.split_off(pivot_index + 1);
                    left.children = root.children.split_off(0);
                }

                // persist
                let left_id = self.arena.insert(left);
                let right_id = self.arena.insert(right);
                // update pointers
                let (left, right) = self.arena.get2_mut(left_id, right_id);
                left.unwrap().right = Some(right_id);
                right.unwrap().left = Some(left_id);

                // update children in parent
                root.children.push(left_id);
                root.children.push(right_id);

                // update root
                self.arena.insert_at(child_id, root);
            }
            // no further balancing is needed
            return;
        }

        let parent_id = parent_id.unwrap();
        let child_index = child_index.unwrap();
        let mut parent = self.arena.get(parent_id).unwrap().clone();
        // load the siblings ordered from left to right
        let (mut siblings, divider_key_start) = self.get_siblings(&parent, child_index);

        if child_is_leaf {
            // copy keys and values
            let mut keys_and_values: VecDeque<_> = VecDeque::new();
            for i in 0..siblings.len() {
                let sibling = siblings.get_mut(i).unwrap();
                for (key, value) in sibling.node.keys.drain(..).zip(sibling.node.values.drain(..)) {
                    keys_and_values.push_back((key, value));
                }

                // remove the divider key if this is not the last sibling
                if i < siblings.len() - 1 {
                    parent.keys.remove(divider_key_start);
                }
            }

            // distribute
            let mut insert_divider_key_at = divider_key_start;
            for sibling in siblings.iter_mut() {
                // fill
                for _ in 0..(self.b * 2 - 1) {
                    if let Some((key, value)) = keys_and_values.pop_front() {
                        sibling.node.keys.push(key);
                        sibling.node.values.push(value);
                    } else {
                        // filled all key values
                        break;
                    }
                }

                // we will reach here when we completely fill the current sibling node
                // if there are still keys left then add divider key
                if !keys_and_values.is_empty() {
                    parent.keys.insert(insert_divider_key_at, sibling.node.keys.last().unwrap().clone());
                    insert_divider_key_at += 1;
                } else {
                    break;
                }
            }

            // if we still have keys and values left add those inside new sibling and adjust pointers
            if !keys_and_values.is_empty() {
                let mut new_sibling = BPlusTreeNode::<K, V>::new();
                while let Some((key, value)) = keys_and_values.pop_front() {
                    new_sibling.keys.push(key);
                    new_sibling.values.push(value);
                }
                assert!(new_sibling.keys.len() <= 2 * self.b - 1, "new sibling node overflow");
                // update pointers
                let Sibling {id: current_rightmost_node_id, node: current_rightmost_node} = siblings.last_mut().unwrap();
                // new sibling node sits in the middle of current rightmost and one node after that (called as next below)
                new_sibling.right = current_rightmost_node.right;
                new_sibling.left = Some(*current_rightmost_node_id);
                let new_sibling_id = self.arena.insert(new_sibling);
                // update left for next node after new sibling
                if let Some(next) = current_rightmost_node.right {
                    let next_node = self.arena.get_mut(next).unwrap();
                    next_node.left = Some(new_sibling_id);
                }
                current_rightmost_node.right = Some(new_sibling_id);
                // add new sibling in parent
                // new sibling will be right to the last divider key that we've added
                parent.children.insert(insert_divider_key_at, new_sibling_id);
            } else {
                if let Some(first_empty_sibling_index) = siblings.iter().position(|s| s.node.keys.is_empty()) {
                    let next = siblings.last_mut().unwrap().node.right;
                    let last_filled_sibling = siblings.get_mut(first_empty_sibling_index - 1).unwrap();
                    // update right for last filled
                    last_filled_sibling.node.right = next;
                    // update left for next node after last filled
                    if let Some(next) = next {
                        let next_node = self.arena.get_mut(next).unwrap();
                        next_node.left = Some(last_filled_sibling.id);
                    }
                    // remove empty nodes from siblings list, arena and from parent
                    for Sibling {id, ..} in siblings.drain(first_empty_sibling_index..) {
                        parent.children.remove(parent.children.iter().position(|child_id| *child_id == id).unwrap());
                        self.arena.remove(id);
                    }
                }
            }

            // assert pointers
            // @note: this can be made conditional
            for (sibling_index, Sibling {id, node}) in siblings.iter().enumerate() {
                if sibling_index + 1 < siblings.len() {
                    let Sibling {node: right, id: right_id} = siblings.get(sibling_index + 1).unwrap();
                    assert_eq!(node.right.unwrap(), *right_id, "Incorrect sibling pointers");
                    assert_eq!(right.left.unwrap(), *id, "Incorrect sibling pointers");
                } else {
                    if let Some(right_id) = node.right {
                        let right = self.arena.get(right_id).unwrap();
                        assert_eq!(right.left.unwrap(), *id, "Incorrect sibling pointers");
                    }
                }
            }
        } else {
            // copy keys and children
            let mut keys: VecDeque<_> = VecDeque::new();
            let mut children: VecDeque<_> = VecDeque::new();
            for i in 0..siblings.len() {
                let sibling = siblings.get_mut(i).unwrap();
                for key in sibling.node.keys.drain(..) { keys.push_back(key); }
                for child in sibling.node.children.drain(..) { children.push_back(child); }
                // copy the divider key if this is not the last sibling
                if i < siblings.len() - 1 {
                    keys.push_back(parent.keys.remove(divider_key_start));
                }
            }

            // distribute
            let mut insert_divider_key_at = divider_key_start;
            let mut current_sibling = 0;
            let mut remaining_keys = keys.len();

            while remaining_keys > 0 && current_sibling < siblings.len() {
                let sibling = siblings.get_mut(current_sibling).unwrap();
                // if exactly two keys are remaining and this node does not have enough space
                // one key will get added inside parent and other inside the next sibling
                if remaining_keys == 2 && sibling.node.keys.len() + 2 > self.b * 2 - 1 {
                    // attach right child
                    sibling.node.children.push(children.pop_front().unwrap());
                    // insert divider key
                    parent.keys.insert(insert_divider_key_at, keys.pop_front().unwrap());
                    remaining_keys -= 1;
                    insert_divider_key_at += 1;
                    // remaining keys and children will be pushed inside new sibling
                    current_sibling += 1;
                } else {
                    // add key and left child
                    sibling.node.keys.push(keys.pop_front().unwrap());
                    remaining_keys -= 1;
                    sibling.node.children.push(children.pop_front().unwrap());

                    // if this sibling gets full
                    if sibling.node.keys.len() == self.b * 2 - 1 {
                        // attach right child
                        sibling.node.children.push(children.pop_front().unwrap());
                        // remaining keys and children will be pushed inside new sibling
                        current_sibling += 1;
                        // if we have some remaining keys then divider needs to get added
                        if remaining_keys > 0 {
                            // insert divider key
                            parent.keys.insert(insert_divider_key_at, keys.pop_front().unwrap());
                            remaining_keys -= 1;
                            insert_divider_key_at += 1;
                        }
                    }
                    // else if sibling does not get full but all keys are distributed
                    else if remaining_keys == 0 {
                        // attach right child
                        sibling.node.children.push(children.pop_front().unwrap());
                        break;
                    }
                }
            }

            // all siblings are filled but we still have remaining keys
            if remaining_keys > 0 {
                let mut new_sibling = BPlusTreeNode::<K, V>::new();
                // add all remaining keys and children
                new_sibling.keys = keys.into_iter().collect();
                new_sibling.children = children.into_iter().collect();
                assert!(new_sibling.keys.len() <= 2 * self.b - 1, "new new_sibling node overflow");
                // persist new sibling
                let new_sibling_id = self.arena.insert(new_sibling);
                // add new sibling in parent
                // new sibling will be right to the last divider key that we've added
                parent.children.insert(insert_divider_key_at, new_sibling_id);
            }
            // if we have some empty siblings
            else if let Some(first_empty_sibling_index) = siblings.iter().position(|s| s.node.keys.is_empty()) {
                 // remove empty nodes from siblings list, arena and corresponding child_id's form parent
                 for Sibling {id, ..} in siblings.drain(first_empty_sibling_index..) {
                    parent.children.remove(parent.children.iter().position(|child_id| *child_id == id).unwrap());
                    self.arena.remove(id);
                }
            }
        }

        //
        // persist updated siblings and parent
        //
        for Sibling {id, node} in siblings {
            self.arena.insert_at(id, node);
        }
        self.arena.insert_at(parent_id, parent);

        if let Some(path_detail) = path.pop() {
            self.balance(Some(path_detail.parent_id), path_detail.child_id, Some(path_detail.child_index), path);
        } else {
            self.balance(None, parent_id, None, path);
        }
    }

    pub fn print(&self) {
        println!("----------------------------------------------------------------");

        struct LevelAndNode {
            level: i32,
            node_id: ArenaNodeIndex,
        }

        if self.root.is_none() {
            println!("                          empty                          ");
            return;
        }

        let mut curr_level = 1;
        let mut nodes = VecDeque::new();
        nodes.push_back(LevelAndNode {level: curr_level, node_id: self.root.unwrap()});
        let mut curr;

        let mut curr_level_text = String::new();
        while !nodes.is_empty() {
            curr = nodes.pop_front().unwrap();

            if curr.level > curr_level {
                println!("L{curr_level}:  {curr_level_text}");
                curr_level += 1;
                curr_level_text = String::new();
            }

            let node = self.arena.get(curr.node_id).expect(
                format!("Error while getting nodeId: {:?} it is not present in arena but is linked as child node", curr.node_id).as_str()
            );
            curr_level_text.push_str(format!("  {:?}", node.keys).as_str());
            //curr_level_text.push_str(format!("  {:?}", node).as_str());

            if !node.children.is_empty() {
                assert_eq!(node.children.len(), node.keys.len() + 1,  "tree is not balanced");
            }

            for child in &node.children {
                nodes.push_back(LevelAndNode {level: curr_level + 1, node_id: *child});
            }
        }

        println!("L{curr_level}:  {curr_level_text}");
        println!("----------------------------------------------------------------");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{seq::SliceRandom};

    #[test]
    fn incremental_inserts_and_deletes() {
        let mut t: BPlusTree<u32, u32> = BPlusTree::new(5);
        for i in 1..=1000 {
            assert_eq!(t.insert(i, i), None);
            assert_eq!(t.get(&i), Some(&i));
            assert_eq!(t.remove(&i), Some(i));
            assert_eq!(t.get(&i), None);
        }
    }

    #[test]
    fn decremental_inserts_and_deletes() {
        let mut t: BPlusTree<u32, u32> = BPlusTree::new(3);
        for i in (1..=1000).rev() {
            assert_eq!(t.insert(i, i), None);
            assert_eq!(t.get(&i), Some(&i));
            assert_eq!(t.remove(&i), Some(i));
            assert_eq!(t.get(&i), None);
        }
    }

    #[test]
    fn incremental_inserts_and_random_deletes() {
        let mut t: BPlusTree<u32, u32> = BPlusTree::new(6);
        let mut array: [u32; 1000] = [0; 1000];
        for (i, elem) in array.iter_mut().enumerate() {
            *elem = (i + 1) as u32;
        }
        for i in array {
            assert_eq!(t.insert(i, i), None);
        }
        array.shuffle(&mut rand::rng());
        for i in array {
            assert_eq!(t.get(&i), Some(&i));
            assert_eq!(t.remove(&i), Some(i));
            assert_eq!(t.get(&i), None);
        }
    }

    #[test]
    fn incremental_inserts_and_decremental_deletes() {
        let mut t: BPlusTree<u32, u32> = BPlusTree::new(4);
        for i in 1..=1000 {
            assert_eq!(t.insert(i, i), None);
        }
        for i in (1..=1000).rev() {
            assert_eq!(t.get(&i), Some(&i));
            assert_eq!(t.remove(&i), Some(i));
            assert_eq!(t.get(&i), None);
        }
    }

    #[test]
    fn random_inserts_and_deletes() {
        for _ in 0..10 {
            let mut t: BPlusTree<u32, u32> = BPlusTree::new(3);
            let mut array: [u32; 1000] = [0; 1000];
            for (i, elem) in array.iter_mut().enumerate() {
                *elem = (i + 1) as u32;
            }
            array.shuffle(&mut rand::rng());
            for i in array {
                assert_eq!(t.insert(i, i), None);
            }
            array.shuffle(&mut rand::rng());
            t.print();
            for i in array {
                assert_eq!(t.get(&i), Some(&i));
                assert_eq!(t.remove(&i), Some(i));
                assert_eq!(t.get(&i), None);
            }
        }
    }

    #[test]
    fn nodes_are_freed_from_arena_after_deletion() {
        let mut t = BPlusTree::<u32, u32>::new(2);
        assert_eq!(t.arena.iter().len(), 0);
        t.insert(1, 1);
        t.insert(2, 2);
        t.insert(3, 3);
        assert_eq!(t.arena.iter().len(), 1);
        t.insert(4, 4);
        t.insert(5, 5);
        assert_eq!(t.arena.iter().len(), 3);
        t.insert(6, 6);
        t.insert(7, 7);
        t.insert(8, 8);
        t.insert(9, 9);
        t.insert(10, 10);
        t.insert(11, 11);
        t.insert(12, 12);
        t.insert(13, 13);
        assert_eq!(t.arena.iter().len(), 8);
        t.remove(&10);
        t.remove(&11);
        t.remove(&2);
        t.remove(&3);
        t.remove(&4);
        t.remove(&5);
        t.remove(&7);
        t.remove(&8);
        t.remove(&9);
        assert_eq!(t.arena.iter().len(), 7);
        t.remove(&12);
        assert_eq!(t.arena.iter().len(), 4);
        t.remove(&1);
        assert_eq!(t.arena.iter().len(), 1);
        t.remove(&6);
        assert_eq!(t.arena.iter().len(), 1);
        t.remove(&13);
        assert_eq!(t.arena.iter().len(), 0);
    }

    #[test]
    fn test_get_in_range() {
        let mut t: BPlusTree<u32, u32> = BPlusTree::new(5);
        for i in 1..=100 { t.insert(i, i); }
        let values = t.get_in_range(&1, &100);
        for i in 1..=100 {
            assert_eq!(*values.get(i - 1).unwrap(), &(i as u32));
        }
    }
}
