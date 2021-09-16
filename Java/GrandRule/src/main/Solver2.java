package main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

public class Solver2 {
	final static String cvsSplitBy = ";";
	private List<Classification> classifications;

	public Solver2(String filename) {
		String line = "";
		ArrayList<Rule> rules = new ArrayList<>(20000);
		classifications = new ArrayList<>(50);

		try (BufferedReader br = new BufferedReader(new FileReader(filename))) {

			while ((line = br.readLine()) != null) {

				rules.add(new Rule(line.split(cvsSplitBy)));
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		Collections.sort(rules);

		Classification c = new Classification(rules.get(0).getClassification());
		c.addRule(rules.get(0));
		classifications.add(c);
		HashSet<Rule> map = new HashSet<Rule>();

		for (int i = 1; i < rules.size(); i++) {
			if (rules.get(i).getClassification() != c.getClassification()) {
				map.clear();
				c = new Classification(rules.get(i).getClassification());
				c.addRule(rules.get(i));
				map.add(rules.get(i));
				classifications.add(c);
			} else {
				if(map.add(rules.get(i))) {
					c.addRule(rules.get(i));
				}
			}
		}
		rules = null;
		System.out.println("Rules loaded...");
	}

	public void solve(String input, String output) {
		for(Classification c: this.classifications) {
			System.out.println("Size of class " + c.getClassification() + ": "+ c.getSize());
		}
		String line;
		ArrayList<Transaction> transactions = new ArrayList<>(10000);

		try (BufferedReader br = new BufferedReader(new FileReader(input))) {

			while ((line = br.readLine()) != null) {

				transactions.add(new Transaction(line.split(cvsSplitBy)));
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		File csvOutputFile = new File(output);
		int nTransactions = 0;

		long start = System.currentTimeMillis();
		try (PrintWriter pw = new PrintWriter(csvOutputFile)) {
			for (Transaction t : transactions) {
				for (Classification cl : classifications) {
					if (cl.matches(t)) {
						pw.println(t.toString() + cl.getClassification());
					}
				}
				nTransactions++;
			}
			double timeTaken = (System.currentTimeMillis()-start)/1000.0;
			System.out.println(nTransactions/timeTaken + " Transacions per second");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
