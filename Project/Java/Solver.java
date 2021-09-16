import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Solver {

  static final String cvsSplitBy = ";";
  private List<Classification> classifications;

  public Solver(String filename) {
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

    writeToFile(rules);

    Classification c = new Classification(rules.get(0).getClassification());
    c.addRule(rules.get(0));
    classifications.add(c);

    for (int i = 1; i < rules.size(); i++) {
      if (rules.get(i).getClassification() != c.getClassification()) {
        c = new Classification(rules.get(i).getClassification());
        c.addRule(rules.get(i));
        classifications.add(c);
      } else {
        c.addRule(rules.get(i));
      }
    }
    rules = null;
    System.out.println("Rules loaded...");
  }

  public void writeToFile(List<Rule> rules){
    File out = new File("teste.txt");
    try (PrintWriter pw = new PrintWriter(out)){
      for(Rule r : rules){
        pw.println(r.toString());
      }
    } catch (Exception e) {
      //TODO: handle exception
    }
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

    long start = System.currentTimeMillis();
    try (PrintWriter pw = new PrintWriter(csvOutputFile)) {
      for (Transaction t : transactions) {
        for (Classification cl : classifications) {
          if (cl.matches(t)) {
            pw.println(t.toString() + cl.getClassification());
          }
        }
      }
      double timeTaken = (System.currentTimeMillis() - start) / 1000.0;
      System.out.println(transactions.size() / timeTaken + " Transacions per second");
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }
}
