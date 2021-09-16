import java.util.ArrayList;

public class Classification {

  private int classification;
  private ArrayList<Rule> rules;

  public Classification(int classification) {
    this.classification = classification;
    rules = new ArrayList<>();
  }

  public void addRule(Rule r) {
    if (!rules.contains(r)) {
      rules.add(r);
    }
  }

  public int getClassification() {
    return classification;
  }

  public boolean matches(Transaction t) {
    //		System.out.println("-----------------------------------------------");
    //		System.out.println("Class: " + classification);
    for (Rule rule : rules) {
      //			System.out.println(rule.toString());
      //			System.out.println(t.toString());
      if (rule.matches(t)) {
        //				System.out.println("Accept");
        //				System.out.println("-----------------------------------------------");
        return true;
      }
    }
    //		System.out.println("Reject");
    //		System.out.println("-----------------------------------------------");
    return false;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("Classification: " + classification + "\n");
    for (Rule r : rules) {
      sb.append(r.toString() + "\n");
    }
    return sb.toString();
  }
}
