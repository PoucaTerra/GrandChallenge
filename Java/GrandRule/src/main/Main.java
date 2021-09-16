package main;

import java.util.List;

public class Main {

	final static String INPUTFOLDER = "..\\..\\Input\\";
	final static String OUTPUTFOLDER = "..\\..\\Output\\";

	final static String RULES = "rule_2M.csv";
	final static String TINYRULES = "rule_tiny.csv";	

	static List<Classification> classifications;

	public static void main(String[] args) {

		long time = System.currentTimeMillis();

		//		Solver s = new Solver(INPUTFOLDER+TINYRULES);
		//		s.solve(INPUTFOLDER+"transactions_tiny.csv", OUTPUTFOLDER+"tiny_output.csv");

		Solver2 s = new Solver2(INPUTFOLDER+RULES);
		s.solve(INPUTFOLDER+"transactions_0.csv", OUTPUTFOLDER+"output_0_Solver2.csv");
		System.out.println("Time: " + (System.currentTimeMillis()-time)/1000.0 + "s");
	}
}