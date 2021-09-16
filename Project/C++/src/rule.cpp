#include <string>
using namespace std;

class rule
{
private:
    int type;
    int stars;
    int tuple[10];

public:
    rule(vector<string> line)
    {
        stars = 0;
        type = stoi(line[10]);

        for (int i = 0; i < 10; i++)
        {
            if (line[i].compare("*") == 0)
            {
                tuple[i] = -1;
                stars++;
            }
            else
            {
                tuple[i] = stoi(line[i]);
            }
        }
    }

    int getClassification()
    {
        return type;
    }

    bool matches(transaction t)
    {
        int *a = t.getTuple();
        for (int i = 0; i < 10; ++i)
        {
            if (tuple[i] == -1)
            {
                continue;
            }
            if (tuple[i] != a[i])
            {
                return false;
            }
        }
        return true;
    }

    int compareTo(rule r)
    {
        int n = type - r.type;
        return n == 0 ? r.stars - stars : n;
    }

    string toString()
    {
        string res;
        for (int i = 0; i < 10; i++)
        {
            if (tuple[i] == -1)
            {
                res.append("*;");
            }
            else
            {
                res.append(tuple[i] + ";");
            }
        }
        res.append(to_string(type));
        return res;
    }
};

bool compareRule(rule r1, rule r2)
{
    return r1.compareTo(r2) < 0;
}