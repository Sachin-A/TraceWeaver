#include <iostream>
#include <fstream>
#include <jsoncpp/json/json.h>

using namespace std;
int main(int argc, char** argv)
{
    string data_directory = "/home/sachina3/Projects/disttrace/data/alibaba-data/";

    for (int i = 0; i <= int(argv[1]); i++) {
        string cg_directory = "call-graph-" + to_string(i);
        cout << cg_directory;
    }
    // ifstream ifs(data_directory + );
    // Json::Reader reader;
    // Json::Value obj;
    // reader.parse(ifs, obj);
    // cout << obj["data"] << endl;
    // cout << "Done\n";
    return 0;
}
