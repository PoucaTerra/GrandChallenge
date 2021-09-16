package main;

import java.util.Arrays;

public class Rule implements Comparable<Rule>{

	private int classification;
	private int stars;
	private int[] tuple;

	public Rule(String[] line) {
		this.stars = 0;
		this.tuple = new int[10];

		this.classification = Integer.valueOf(line[10]);

		for(int i = 0; i < 10; i++) {
			if(line[i].equals("*")) {
				this.tuple[i] = -1;
				this.stars++;
			}else {
				this.tuple[i] = Integer.valueOf(line[i]);
			}
		}
	}

	public int getClassification() {
		return classification;
	}
	
	public int getPosition(int pos) {
		return tuple[pos];
	}

	public boolean matches(Transaction t) {

		int[] a = t.getTuple();
		for(int i = 0; i < 10; i++) {
			if(tuple[i] == -1) {
				continue;
			}
			if(tuple[i] != a[i]) {
				return false;
			}
		}
		return true;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(tuple);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Rule other = (Rule) obj;
		if (!Arrays.equals(tuple, other.tuple))
			return false;
		if(other.classification != this.classification)
			return false;
		return true;
	}

	@Override
	public int compareTo(Rule r) {
		int n = classification - r.classification;
		return n == 0 ? r.stars - stars : n;
	}

	@Override
	public String toString() {
		return Arrays.toString(tuple).replace("-1", "*");

	}

}


