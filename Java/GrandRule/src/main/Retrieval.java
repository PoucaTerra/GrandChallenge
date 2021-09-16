package main;

import java.util.HashMap;
import java.util.HashSet;

public class Retrieval {

	private TrieNode root;  

	public Retrieval() {
		this.root = new TrieNode(0);
	}

	public void insert(Rule r) {
		root.insert(r);
	}

	public HashSet<Integer> search(Transaction t){
		return root.search(t);
	}

	private static class TrieNode { 

		private HashMap<Integer,TrieNode> children;
		private HashSet<Integer> classifications;
		private int level;

		TrieNode(int level){ 
			this.level = level;
			if(level == 10) {
				classifications = new HashSet<Integer>();
			}else {
				children = new HashMap<>();
			}
		} 

		// If not present, inserts key into trie 
		// If the key is prefix of trie node,  
		// just marks leaf node 
		public void insert(Rule r) { 

			if(level == 10) {
				classifications.add(r.getClassification());
			}else {
				TrieNode temp = children.putIfAbsent(r.getPosition(level), new TrieNode(this.level+1));
				if(temp == null) {
					temp = children.get(r.getPosition(level));
				}
				temp.insert(r);
			}
		} 

		public HashSet<Integer> search(Transaction t) { 

			if(level == 10) {
				return classifications;
			}else {
				HashSet<Integer> result = new HashSet<Integer>();
				TrieNode temp = children.get(t.getPosition(level));

				if(temp != null) {
					result.addAll(temp.search(t));
				}

				temp = children.get(-1);

				if(temp != null) {
					result.addAll(temp.search(t));
				}
				return result;
			}
		}
	} 
} 