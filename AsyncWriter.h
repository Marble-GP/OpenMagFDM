#ifndef OPENMAGFDM_ASYNC_WRITER_H
#define OPENMAGFDM_ASYNC_WRITER_H

// Single-worker async writer for transient-step output.
//
// Rationale: file writes (CSV / TIFF) dominate step time on fast solver paths.
// Moving them to a worker thread lets the solver overlap I/O with the next
// step's computation. A single worker (not a pool) preserves ordering, which
// matters for live WebUI displays that watch the output directory.
//
// Bounded queue: enqueue blocks once max_depth items are pending, so the
// solver never accumulates unbounded backlog if the disk is slow.
//
// Errors: the first exception from the worker is captured and re-thrown from
// the next enqueue() / drain() call, so failures propagate to the solver
// thread without being silently swallowed.

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

class AsyncWriter {
public:
    // Each queued job may own a full field matrix. Keep configuration errors
    // from turning into an unexpectedly large memory backlog.
    static constexpr std::size_t kMaxQueueDepth = 16;

    explicit AsyncWriter(std::size_t max_depth);
    ~AsyncWriter();

    AsyncWriter(const AsyncWriter&) = delete;
    AsyncWriter& operator=(const AsyncWriter&) = delete;

    // Submit a job. Blocks if the queue is at max_depth so the producer
    // cannot run away from the disk.
    void enqueue(std::function<void()> job);

    // Block until every queued job has finished. Re-throws the first
    // worker-side exception if any occurred.
    void drain();

private:
    void workerLoop();
    void rethrowIfFailedLocked(std::unique_lock<std::mutex>& lock);

    std::size_t max_depth_;
    std::mutex mu_;
    std::condition_variable cv_not_full_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_idle_;
    std::queue<std::function<void()>> jobs_;
    std::size_t in_flight_ = 0;  // jobs currently being executed (0 or 1 with single worker)
    bool stop_ = false;
    std::exception_ptr first_error_ = nullptr;
    std::thread worker_;
};

#endif  // OPENMAGFDM_ASYNC_WRITER_H
