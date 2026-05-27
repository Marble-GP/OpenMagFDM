#include "AsyncWriter.h"

AsyncWriter::AsyncWriter(std::size_t max_depth)
    : max_depth_(max_depth == 0 ? 1 : max_depth) {
    worker_ = std::thread(&AsyncWriter::workerLoop, this);
}

AsyncWriter::~AsyncWriter() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_not_empty_.notify_all();
    cv_not_full_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void AsyncWriter::enqueue(std::function<void()> job) {
    std::unique_lock<std::mutex> lk(mu_);
    rethrowIfFailedLocked(lk);
    cv_not_full_.wait(lk, [this] { return stop_ || jobs_.size() < max_depth_; });
    if (stop_) return;
    jobs_.push(std::move(job));
    lk.unlock();
    cv_not_empty_.notify_one();
}

void AsyncWriter::drain() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_idle_.wait(lk, [this] { return stop_ || (jobs_.empty() && in_flight_ == 0); });
    rethrowIfFailedLocked(lk);
}

void AsyncWriter::workerLoop() {
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_not_empty_.wait(lk, [this] { return stop_ || !jobs_.empty(); });
            if (stop_ && jobs_.empty()) {
                cv_idle_.notify_all();
                return;
            }
            job = std::move(jobs_.front());
            jobs_.pop();
            in_flight_ = 1;
        }
        cv_not_full_.notify_one();

        try {
            job();
        } catch (...) {
            std::lock_guard<std::mutex> lk(mu_);
            if (!first_error_) {
                first_error_ = std::current_exception();
            }
        }

        {
            std::lock_guard<std::mutex> lk(mu_);
            in_flight_ = 0;
        }
        cv_idle_.notify_all();
    }
}

void AsyncWriter::rethrowIfFailedLocked(std::unique_lock<std::mutex>& /*lock*/) {
    if (first_error_) {
        auto err = first_error_;
        first_error_ = nullptr;
        std::rethrow_exception(err);
    }
}
