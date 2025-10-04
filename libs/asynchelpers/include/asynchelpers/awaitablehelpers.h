//
// Created by Francisco Nunez on 28.09.2025.
//

#ifndef CURVEFORGE_AWAITABLEHELPERS_H
#define CURVEFORGE_AWAITABLEHELPERS_H
#pragma once
#include <coroutine>
#include <exception>
#include <functional>
#include <optional>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

//==================== task<T> (awaitable result) ====================//
template<class T>
struct task {
    struct promise_type {
        std::optional<T> value;
        std::exception_ptr eptr;
        std::coroutine_handle<> continuation{};

        task get_return_object() {
            return task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        // Lazy: start when awaited
        std::suspend_always initial_suspend() noexcept { return {}; }

        // Resume whoever awaited us once we're done
        struct final_awaitable {
            bool await_ready() noexcept { return false; }

            void await_suspend(std::coroutine_handle<promise_type> h) noexcept {
                if (auto c = h.promise().continuation) c.resume();
            }

            void await_resume() noexcept {
            }
        };

        final_awaitable final_suspend() noexcept { return {}; }

        void return_value(T v) noexcept(std::is_nothrow_move_constructible_v<T>) { value = std::move(v); }
        void unhandled_exception() { eptr = std::current_exception(); }
    };

    using handle = std::coroutine_handle<promise_type>;
    handle h{};

    explicit task(handle hh) : h(hh) {
    }

    task(task &&o) noexcept : h(std::exchange(o.h, {})) {
    }

    task(const task &) = delete;

    task &operator=(task &&o) noexcept {
        if (this != &o) {
            if (h) h.destroy();
            h = std::exchange(o.h, {});
        }
        return *this;
    }

    task &operator=(const task &) = delete;

    ~task() { if (h) h.destroy(); }

    struct awaiter {
        handle h;
        bool await_ready() const noexcept { return !h || h.done(); }

        void await_suspend(std::coroutine_handle<> cont) {
            h.promise().continuation = cont;
            h.resume(); // start/continue the task
        }

        T await_resume() {
            if (h.promise().eptr) std::rethrow_exception(h.promise().eptr);
            return std::move(*h.promise().value);
        }
    };

    auto operator co_await() { return awaiter{h}; }
};

//==================== task<void> specialization ====================//
template<>
struct task<void> {
    struct promise_type {
        std::exception_ptr eptr;
        std::coroutine_handle<> continuation{};

        task get_return_object() {
            return task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }

        struct final_awaitable {
            bool await_ready() noexcept { return false; }

            void await_suspend(std::coroutine_handle<promise_type> h) noexcept {
                if (auto c = h.promise().continuation) c.resume();
            }

            void await_resume() noexcept {
            }
        };

        final_awaitable final_suspend() noexcept { return {}; }

        void return_void() noexcept {
        }

        void unhandled_exception() { eptr = std::current_exception(); }
    };

    using handle = std::coroutine_handle<promise_type>;
    handle h{};

    explicit task(handle hh) : h(hh) {
    }

    task(task &&o) noexcept : h(std::exchange(o.h, {})) {
    }

    task(const task &) = delete;

    task &operator=(task &&o) noexcept {
        if (this != &o) {
            if (h) h.destroy();
            h = std::exchange(o.h, {});
        }
        return *this;
    }

    task &operator=(const task &) = delete;

    ~task() { if (h) h.destroy(); }

    struct awaiter {
        handle h;
        bool await_ready() const noexcept { return !h || h.done(); }

        void await_suspend(std::coroutine_handle<> cont) {
            h.promise().continuation = cont;
            h.resume();
        }

        void await_resume() {
            if (h.promise().eptr) std::rethrow_exception(h.promise().eptr);
        }
    };

    auto operator co_await() { return awaiter{h}; }
};

//==================== background_awaitable<T> ====================//
// Runs a callable on a background jthread and resumes the awaiting coroutine.
template<class T, class Callable>
struct background_awaitable {
    Callable fn;
    std::optional<T> result;
    std::exception_ptr eptr;

    bool await_ready() const noexcept { return false; }

    void await_suspend(std::coroutine_handle<> h) {
        std::jthread([this, h]() mutable {
            try {
                if constexpr (std::is_void_v<T>) {
                    std::invoke(fn);
                } else {
                    result = std::invoke(fn);
                }
            } catch (...) {
                eptr = std::current_exception();
            }
            h.resume();
        }).detach();
    }

    T await_resume() {
        if (eptr) std::rethrow_exception(eptr);
        if constexpr (!std::is_void_v<T>) return std::move(*result);
    }
};

// Specialization for void
//==================== background_awaitable<void, Callable> ====================//
template<class Callable>
struct background_awaitable<void, Callable> {
    Callable fn;
    std::exception_ptr eptr;

    bool await_ready() const noexcept { return false; }

    void await_suspend(std::coroutine_handle<> h) {
        std::jthread([this, h]() mutable {
            try {
                std::invoke(fn);
            } catch (...) {
                eptr = std::current_exception();
            }
            h.resume();
        }).detach();
    }

    void await_resume() {
        if (eptr) std::rethrow_exception(eptr);
    }
};

//==================== async_call(F, Args...) -> task<T> ====================//
template<class F, class... Args>
auto async_call(F &&f, Args &&... args)
    -> task<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...> > {
    using T = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
    auto bound = [ff = std::forward<F>(f),
                tup = std::make_tuple(std::forward<Args>(args)...)]() mutable -> T {
        return std::apply(std::move(ff), std::move(tup));
    };
    if constexpr (std::is_void_v<T>) {
        co_await background_awaitable<void, decltype(bound)>{std::move(bound)};
        co_return;
    } else {
        T v = co_await background_awaitable<T, decltype(bound)>{std::move(bound)};
        co_return std::move(v);
    }
}
#endif //CURVEFORGE_AWAITABLEHELPERS_H
