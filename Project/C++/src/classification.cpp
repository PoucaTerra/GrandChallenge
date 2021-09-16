#include <vector>
#include <algorithm>
using namespace std;

bool findRule(vector<rule> v, rule r)
{
    for (int i = 0; i < v.size(); i++)
    {
        if (v[i].compareTo(r) == 0)
        {
            return true;
        }
    }
    return false;
}

class classification
{
private:
    int type;
    vector<rule> rules;

public:
    classification(int c)
    {
        type = c;
    }

    void addRule(rule r)
    {

        if (!findRule(rules, r))
        {
            rules.push_back(r);
        }
    }

    int getClassification()
    {
        return type;
    }

    bool matches(transaction t)
    {
        for (int i = 0; i < rules.size(); i++)
        {
            if (rules[i].matches(t))
            {
                return true;
            }
        }
        return false;
    }

    string toString()
    {
        string res = "";
        res.append("Classification: " + to_string(type) + "\n");
        for (int i = 0; i < rules.size(); i++)
        {
            res.append(rules[i].toString() + "\n");
        }
        return res;
    }
};