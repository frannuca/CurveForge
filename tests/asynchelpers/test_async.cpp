//
// Created by Francisco Nunez on 28.09.2025.
//
// demo.cpp
#include "asynchelpers/awaitablehelpers.h"
#include <chrono>
#include <coroutine>
#include <iostream>
#include <thread>

using namespace std::chrono_literals;

// A "top-level" runner that starts immediately (no event loop here)
struct runner {
    struct promise_type {
        runner get_return_object() { return {}; }
        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }

        void return_void() noexcept {
        }

        void unhandled_exception() { std::terminate(); }
    };
};

runner demo() {
    // Any return type works (int, std::string, custom types, void…)
    int sum = co_await async_call([](int a, int b) {
        std::this_thread::sleep_for(150ms);
        return a + b;
    }, 20, 22);

    std::cout << "sum = " << sum << "\n";

    std::string s = co_await async_call([] {
        std::this_thread::sleep_for(100ms);
        return std::string("hello from async_call");
    });
    std::cout << s << "\n";

    co_await async_call([] {
        std::this_thread::sleep_for(50ms);
        // returns void
    });
    std::cout << "void task done\n";
}

int main() {
    demo(); // fire coroutines
    std::this_thread::sleep_for(500ms); // keep process alive for demo
}
