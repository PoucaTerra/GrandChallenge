package main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;

public class Solver {

	final static String cvsSplitBy = ";";
	private Retrieval ret;

	public Solver(String filename) {
		ret = new Retrieval();
		String line = "";

		try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
			while ((line = br.readLine()) != null) {
				ret.insert(new Rule(line.split(cvsSplitBy)));
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("Rules loaded...");
	}

	public void solve(String input, String output) {
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
				HashSet<Integer> result = ret.search(t);
				for(Integer i : result) {
					pw.println(t.toString() + i);
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
