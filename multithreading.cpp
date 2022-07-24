//懒汉模式
class singleton{
private:
static singleton* p;
singleton() {};

public:
static singleton* instance();
};
singleton* singleton::instance(){
    if(p == nullptr)
        p = new singleton();
    return p;
}
//饿汉模式
class singleton_th{
private:
singleton_th() {}
~singleton_th(){}
singleton_th(const singleton_th&);
singleton_th& operator=(const singleton_th&);

public:
static singleton_th& instance();
};
//生产者消费者
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace std;
class producer_consumer
{
private:
size_t begin;
size_t end;
size_t cur;
vector<int> buffer;
condition_variable not_full;
condition_variable not_empty;
mutex mut;

public:
//prevent copy
producer_consumer(const producer_consumer& rhs) = delete;
producer_consumer& operator=(const producer_consumer& rhs) = delete;
//init
producer_consumer(size_t sz): begin(0), end(0), cur(0), buffer(sz){}

void produce(int n)
{
    {
        unique_lock<mutex> lock(mut);
        not_full.wait(lock, [=]{return cur < buffer.size();});
        //add new
        buffer[end] = n;
        end = (end + 1) % buffer.size();
        ++cur;
    }
    not_empty.notify_one();
}

int comsume()
{
    unique_lock<mutex> lock(mut);
    not_empty.wait(lock, [=]{return cur > 0;});
    int n = buffer[begin];
    begin = (begin + 1) % buffer.size();
    --cur;
    lock.unlock();
    not_full.notify_one();
    return n;
}   
};

producer_consumer buffers(2);
mutex io_mutex;

void producer()
{
    int n = 0;
    while(n < 10)
    {
        buffers.produce(n);
        unique_lock<mutex> lock(io_mutex);
        cout << "produce ---" << n << endl;
        lock.unlock();
        n++;
    }
    buffers.produce(-1);
}

void comsumer()
{
    thread::id thread_id = this_thread::get_id();
    int n = 0;
    do
    {
        n = buffers.comsume();
        unique_lock<mutex> lock(io_mutex);
        cout << "comsume ---" << n << endl;
        lock.unlock();
    }while (n != -1);
    buffers.produce(-1);
}

int main(int argc, char const *argv[])
{
    vector<thread> threads;
    threads.push_back(thread(&producer));
    threads.push_back(thread(&comsumer));
    threads.push_back(thread(&comsumer));
    threads.push_back(thread(&comsumer));
    for(auto& t : threads)
        t.join();
    return 0;
}

//线程池
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <queue>
#include <memory>
#include <future>
#include <functional>
#include <stdexcept>

class ThreadPool{
public:
    ThreadPool(size_t);
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();
private:
    std::vector<std::thread> workers; 
    std::queue<std::function<void()>> tasks; // task queue
    std::mutex queue_mutex; // synchronization
    std::condition_variable condition;
    bool stop;  
};
inline ThreadPool::ThreadPool(size_t threads) : stop(false)
{
    for(size_t i = 0; i < threads; ++i)
        workers.emplace_back(
            [this]
            {
                for(;;)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, 
                            [this]{return this->stop || !this->tasks.empty();});
                        if(this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            }
        );
}
//add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    ->std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<return_type>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");
        tasks.emplace([task](){(*task)();});
    }
    condition.notify_one();
    return res;
}

inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers){
        worker.join();
    }
}
#endif