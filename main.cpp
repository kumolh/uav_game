#include <iostream>

class singleton{
private:
    singleton(){};
    singleton(const singleton& tmp){}
    singleton& operator=(const singleton& tmp);
    static singleton* p;
public:
    static singleton* getInstance();
    
};

singleton* singleton::getInstance()
{
    return p;
}
singleton* singleton::p = new singleton();

int main(){
    singleton* p = singleton::getInstance();
    singleton* q = singleton::getInstance();
    return 0;
}