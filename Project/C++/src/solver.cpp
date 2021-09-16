#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include "transaction.cpp"
#include "rule.cpp"
#include "classification.cpp"
using namespace std;

class solver
{
private:
    vector<classification> classifications;
public:
    solver(string rulespath)
    {
        ifstream inputFile;
        inputFile.open(rulespath, ios::in);

        string line = "";
        vector<rule> rules;
        
        while (getline(inputFile, line))
        {
            vector<string> temp;
            boost::split(temp, line, [](char c) { return c == ';'; });
            rules.push_back(rule(temp));
        }

        sort(rules.begin(), rules.end(), compareRule);

        classification c = classification(rules[0].getClassification());
        c.addRule(rules[0]);
        classifications.push_back(c);

        for (int i = 1; i < rules.size(); i++)
        {
            if (rules[i].getClassification() != c.getClassification())
            {
                c = classification(rules[i].getClassification());
                c.addRule(rules[i]);
                classifications.push_back(c);
            }
            else
            {
                c.addRule(rules[i]);
            }
        }
        cout << "Completed reading rules..." << endl;
    }

    void solve(string in, string out)
    {

        vector<transaction> transactions;

        ifstream infile;
        infile.open(in, ios::in);

        string line;
        while (getline(infile, line))
        {
            vector<string> temp;
            boost::split(temp, line, [](char c) { return c == ';'; });
            transactions.push_back(transaction(temp));
        }

        ofstream outfile;
        outfile.open(out, ios::out);

        for (int i = 0; i < transactions.size(); i++)
        {
            string s = "";
            for (int j = 0; j < classifications.size(); j++)
            {
                if(classifications[j].matches(transactions[i])){
                    s.append(transactions[i].toString());
                    s.append(to_string(classifications[j].getClassification()));
                    s.append("\n");
                }
            }
            outfile << s;
        }

        infile.close();
        outfile.close();
    }
};