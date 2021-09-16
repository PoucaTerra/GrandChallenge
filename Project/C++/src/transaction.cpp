#include <unordered_set>
#include <string>
using namespace std;

class transaction
{
private:
    int tuple[10];

public:
    transaction(vector<string> line)
    {
        for (int i = 0; i < 10; i++)
        {
            tuple[i] = stoi(line[i]);
        }
    }

    int *getTuple()
    {
        return tuple;
    }

    string toString()
    {
        string res = "";
        for (int i = 0; i < 10; i++)
        {
            res.append(to_string(tuple[i]) + ";");
        }
        return res;
    }
};