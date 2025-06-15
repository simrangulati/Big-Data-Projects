# 🔥 Complete Guide to Python Generator Types – Big Data Edition

A comprehensive exploration of all generator types and patterns supported in Python, with real-world big data scenarios and performance optimizations.

---

## ⚙️ How Yield Works Internally

Understanding the internal mechanics of `yield` is crucial for mastering generators in Python. Let's dive deep into what happens when Python encounters a generator function.

### **🔍 Generator Function vs Regular Function**

```python
# Regular function - returns immediately
def regular_function():
    print("Start")
    return "Hello"
    print("This never executes")

# Generator function - becomes a generator object
def generator_function():
    print("Start")
    yield "Hello"
    print("Middle")
    yield "World"
    print("End")

# Comparison
result = regular_function()          # Executes immediately, prints "Start"
gen = generator_function()          # Returns generator object, nothing printed yet
print(type(gen))                    # <class 'generator'>
```

### **🏗️ Generator Object Lifecycle**

When Python encounters a function with `yield`, it creates a **generator object** instead of executing the function:

```python
import inspect

def sample_generator():
    print("Step 1")
    yield 1
    print("Step 2") 
    yield 2
    print("Step 3")
    return "Done"

# Create generator object
gen = sample_generator()
print(f"Generator state: {inspect.getgeneratorstate(gen)}")  # GEN_CREATED

# First call to next()
value1 = next(gen)  # Prints "Step 1", returns 1
print(f"Value: {value1}")
print(f"Generator state: {inspect.getgeneratorstate(gen)}")  # GEN_SUSPENDED

# Second call to next()
value2 = next(gen)  # Prints "Step 2", returns 2
print(f"Value: {value2}")
print(f"Generator state: {inspect.getgeneratorstate(gen)}")  # GEN_SUSPENDED

# Third call to next() - StopIteration raised
try:
    value3 = next(gen)  # Prints "Step 3", raises StopIteration("Done")
except StopIteration as e:
    print(f"Generator finished with value: {e.value}")
    print(f"Generator state: {inspect.getgeneratorstate(gen)}")  # GEN_CLOSED
```

### **🧠 State Management and Memory**

Python preserves the entire execution state when a generator is suspended:

```python
def stateful_generator():
    # Local variables are preserved across yields
    counter = 0
    data_buffer = []
    
    while counter < 5:
        print(f"Counter: {counter}, Buffer: {data_buffer}")
        
        # Receive input and pause execution
        received = yield counter
        
        # Resume here with preserved state
        if received is not None:
            data_buffer.append(received)
        
        counter += 1
    
    return f"Final buffer: {data_buffer}"

# Demonstrate state preservation
gen = stateful_generator()

# Initialize generator
print("Initializing:")
first_value = next(gen)  # Counter: 0, Buffer: []

# Send data and observe state preservation
print("\nSending 'A':")
second_value = gen.send('A')  # Counter: 1, Buffer: ['A']

print("\nSending 'B':")
third_value = gen.send('B')   # Counter: 2, Buffer: ['A', 'B']

print(f"Values received: {first_value}, {second_value}, {third_value}")
```

### **📚 Stack Frame Preservation**

Python uses a special mechanism to preserve the generator's stack frame:

```python
import sys

def nested_generator():
    def inner_function():
        return "Inner executed"
    
    print("Outer function start")
    local_var = "preserved"
    
    # Stack frame is preserved here
    yield inner_function()
    
    # Local variables still accessible after yield
    print(f"Local variable after yield: {local_var}")
    yield "Second value"

gen = nested_generator()

# Inspect stack frame information
frame = gen.gi_frame
print(f"Frame locals before first next(): {frame.f_locals}")

first = next(gen)  # Prints "Outer function start"
print(f"First yield: {first}")
print(f"Frame locals after first yield: {frame.f_locals}")

second = next(gen)  # Prints "Local variable after yield: preserved"
print(f"Second yield: {second}")
```

### **🔄 Iterator Protocol Implementation**

Generators automatically implement the iterator protocol:

```python
class ManualIterator:
    """Manual implementation of what yield does automatically"""
    def __init__(self):
        self.state = 0
        self.local_vars = {}  # Simulate local variable preservation
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.state == 0:
            self.local_vars['counter'] = 0
            self.state = 1
            return "First"
        elif self.state == 1:
            self.local_vars['counter'] += 1
            self.state = 2
            return f"Second (counter: {self.local_vars['counter']})"
        else:
            raise StopIteration("Done")

# Compare with generator
def auto_generator():
    counter = 0
    yield "First"
    counter += 1
    yield f"Second (counter: {counter})"
    return "Done"

# Both produce the same results
manual = ManualIterator()
auto = auto_generator()

print("Manual iterator:")
for item in manual:
    print(item)

print("\nGenerator (automatic):")  
for item in auto:
    print(item)
```

### **⚡ Performance and Memory Efficiency**

Here's how yield achieves memory efficiency:

```python
import sys
import tracemalloc

def memory_comparison():
    """Compare memory usage of different approaches"""
    
    # Approach 1: Return list (high memory)
    def create_list(n):
        return [x**2 for x in range(n)]
    
    # Approach 2: Generator (low memory)
    def create_generator(n):
        for x in range(n):
            yield x**2
    
    # Approach 3: Generator with state (minimal memory)
    def create_stateful_generator(n):
        current = 0
        while current < n:
            yield current**2
            current += 1
    
    n = 100000
    
    # Measure memory for list
    tracemalloc.start()
    list_data = create_list(n)
    current, peak = tracemalloc.get_traced_memory()
    list_memory = peak
    tracemalloc.stop()
    
    # Measure memory for generator
    tracemalloc.start()
    gen_data = create_generator(n)
    current, peak = tracemalloc.get_traced_memory()
    gen_memory = peak
    tracemalloc.stop()
    
    print(f"List memory usage: {list_memory / 1024 / 1024:.2f} MB")
    print(f"Generator memory usage: {gen_memory / 1024:.2f} KB")
    print(f"Memory savings: {(list_memory - gen_memory) / list_memory * 100:.1f}%")
    
    # Demonstrate lazy evaluation
    print(f"\nList size in memory: {sys.getsizeof(list_data)} bytes")
    print(f"Generator size in memory: {sys.getsizeof(gen_data)} bytes")

memory_comparison()
```

### **🔧 Generator States in Detail**

Python generators have four distinct states:

```python
import inspect

def demonstrate_states():
    yield "First"
    yield "Second" 
    return "Final"

gen = demonstrate_states()

# State 1: GEN_CREATED
print(f"1. Created: {inspect.getgeneratorstate(gen)}")

# State 2: GEN_SUSPENDED (after first yield)
next(gen)
print(f"2. Suspended: {inspect.getgeneratorstate(gen)}")

# State 3: GEN_RUNNING (briefly, during execution)
# We can't easily capture this state as it's transient

# State 4: GEN_CLOSED (after StopIteration)
next(gen)  # Second yield
try:
    next(gen)  # Raises StopIteration
except StopIteration:
    pass
print(f"3. Closed: {inspect.getgeneratorstate(gen)}")

# Additional methods available on generators
print(f"\nGenerator methods:")
print(f"gi_frame: {gen.gi_frame}")  # None when closed
print(f"gi_running: {gen.gi_running}")  # False when not executing
print(f"gi_code: {gen.gi_code.co_name}")  # Function name
```

### **🚀 Advanced: Send, Throw, and Close**

Generators support advanced control flow:

```python
def advanced_generator():
    try:
        print("Generator started")
        received = None
        
        while True:
            print(f"About to yield, last received: {received}")
            received = yield f"Current state: {received}"
            
            if received == "stop":
                break
                
    except GeneratorExit:
        print("Generator is being closed")
        return "Cleanup completed"
    except Exception as e:
        print(f"Exception received: {e}")
        yield f"Handled exception: {e}"
    finally:
        print("Generator cleanup")

gen = advanced_generator()

# Initialize
result1 = next(gen)
print(f"1. {result1}")

# Send values
result2 = gen.send("Hello")
print(f"2. {result2}")

result3 = gen.send("World")  
print(f"3. {result3}")

# Throw exception
try:
    result4 = gen.throw(ValueError("Test error"))
    print(f"4. {result4}")
except StopIteration:
    pass

# Close generator
gen.close()
print("Generator closed")
```

### **🧪 Practical Example: Coroutine Communication**

Here's how yield enables bidirectional communication:

```python
def data_processor():
    """Coroutine that processes data and maintains running statistics"""
    count = 0
    total = 0
    values = []
    
    # Initialize - first yield for priming
    result = yield None
    
    try:
        while True:
            # Receive data
            data = yield {
                'count': count,
                'average': total / count if count > 0 else 0,
                'last_values': values[-5:],  # Last 5 values
                'status': 'processing'
            }
            
            if data is not None:
                count += 1
                total += data
                values.append(data)
                
    except GeneratorExit:
        yield {
            'count': count,
            'average': total / count if count > 0 else 0,
            'all_values': values,
            'status': 'completed'
        }

# Usage example
processor = data_processor()
next(processor)  # Prime the coroutine

# Send data and receive statistics
stats1 = processor.send(10)
print(f"After sending 10: {stats1}")

stats2 = processor.send(20)
print(f"After sending 20: {stats2}")

stats3 = processor.send(30)
print(f"After sending 30: {stats3}")

# Close and get final statistics
processor.close()
```

### **💡 Key Takeaways: How Yield Works**

1. **Function Transformation**: Functions with `yield` become generator factories
2. **Lazy Execution**: Code only runs when `next()` is called
3. **State Preservation**: Local variables and execution position are preserved
4. **Memory Efficiency**: Only current value is kept in memory, not entire sequence
5. **Iterator Protocol**: Generators automatically implement `__iter__()` and `__next__()`
6. **Bidirectional Communication**: Can send values back into generator with `send()`
7. **Exception Handling**: Support for `throw()` and `close()` methods
8. **Stack Frame Magic**: Python preserves the entire execution context

---

## 🧩 Generator Classification Overview

Python supports multiple types of generators, each serving different use cases in data processing and big data applications:

| Generator Type | Syntax | Use Case | Memory Efficiency |
|----------------|--------|----------|------------------|
| **Function Generators** | `def func(): yield x` | Custom logic, complex transformations | ⭐⭐⭐⭐⭐ |
| **Generator Expressions** | `(x for x in data)` | Simple transformations, filtering | ⭐⭐⭐⭐⭐ |
| **Built-in Generators** | `range(), enumerate()` | Standard iteration patterns | ⭐⭐⭐⭐ |
| **Coroutines** | `def func(): x = yield` | Bidirectional communication | ⭐⭐⭐ |
| **Async Generators** | `async def func(): yield x` | Asynchronous data streams | ⭐⭐⭐⭐ |
| **Recursive Generators** | Self-calling generators | Tree/graph traversal | ⭐⭐⭐ |
| **Chained Generators** | Multiple linked generators | Pipeline processing | ⭐⭐⭐⭐⭐ |

---

## 🚀 1. Function Generators (yield keyword)

### **Basic Function Generator**
```python
def simple_generator():
    """Most basic generator function"""
    yield 1
    yield 2
    yield 3

# Usage
gen = simple_generator()
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
```

### **Big Data Scenario: CSV File Processor**
```python
def process_large_csv(filename, chunk_size=1000):
    """Process massive CSV files without loading into memory"""
    import csv
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        batch = []
        
        for row in reader:
            batch.append(row)
            if len(batch) >= chunk_size:
                yield batch
                batch = []
        
        if batch:  # Yield remaining records
            yield batch

# Process 100GB CSV file efficiently
for batch in process_large_csv('massive_data.csv'):
    # Process each batch of 1000 records
    processed_batch = [transform_record(record) for record in batch]
    save_to_database(processed_batch)
```

### **Advanced: Stateful Generator with Memory**
```python
def moving_average_generator(window_size=10):
    """Calculate moving average for streaming data"""
    data_window = []
    
    while True:
        # Receive new value
        new_value = yield
        data_window.append(new_value)
        
        if len(data_window) > window_size:
            data_window.pop(0)
        
        # Calculate and yield moving average
        average = sum(data_window) / len(data_window)
        yield average

# Usage for real-time data processing
avg_gen = moving_average_generator(5)
next(avg_gen)  # Initialize

# Stream processing
for sensor_reading in live_sensor_stream():
    avg_gen.send(sensor_reading)
    current_avg = next(avg_gen)
    if current_avg > threshold:
        trigger_alert()
```

---

## ⚡ 2. Generator Expressions (Comprehension Syntax)

### **Memory Efficient Processing**
```python
# Memory efficient vs list comprehension
numbers = range(1000000)

# ❌ List comprehension - loads all in memory
squares_list = [x**2 for x in numbers]  # ~76MB

# ✅ Generator expression - lazy evaluation
squares_gen = (x**2 for x in numbers)   # ~0.1KB
```

### **Big Data Log Analysis Pipeline**
```python
def analyze_log_files(log_files):
    """Multi-stage generator pipeline for log analysis"""
    
    # Stage 1: Read all log files
    all_lines = (
        line.strip() 
        for filename in log_files 
        for line in open(filename, 'r')
    )
    
    # Stage 2: Filter error lines
    error_lines = (
        line for line in all_lines 
        if 'ERROR' in line or 'CRITICAL' in line
    )
    
    # Stage 3: Parse structured data
    parsed_errors = (
        {
            'timestamp': line[:19],
            'level': 'ERROR' if 'ERROR' in line else 'CRITICAL',
            'message': line[20:],
        }
        for line in error_lines
    )
    
    return parsed_errors
```

### **Conditional Generator Expressions**
```python
# Data quality filtering
def quality_filter(data_stream, min_quality=0.8):
    """Filter data based on quality score"""
    return (
        record for record in data_stream 
        if record.get('quality_score', 0) >= min_quality
    )

# Multi-condition filtering
def sensor_data_filter(sensor_stream):
    """Complex filtering for sensor data"""
    return (
        reading for reading in sensor_stream
        if (
            reading['temperature'] is not None and
            -50 <= reading['temperature'] <= 100 and
            reading['humidity'] is not None and
            0 <= reading['humidity'] <= 100 and
            reading.get('sensor_status') == 'active'
        )
    )
```

---

## 🔧 3. Built-in Generators

### **Range and Enumerate**
```python
# Date range generator
from datetime import datetime, timedelta

def date_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)

# Process daily data for a year
start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)

for date in date_range(start, end):
    daily_data = fetch_data_for_date(date)
    process_daily_data(daily_data)
```

### **Enumerate Generator**
```python
# Built-in enumerate is a generator
def custom_enumerate(iterable, start=0):
    """Custom implementation of enumerate"""
    index = start
    for item in iterable:
        yield index, item
        index += 1

# Big data usage
def process_with_progress(large_dataset):
    total = len(large_dataset)
    for index, record in enumerate(large_dataset):
        # Process record
        process_record(record)
        
        # Show progress every 1000 records
        if index % 1000 == 0:
            progress = (index / total) * 100
            print(f"Progress: {progress:.1f}%")
```

### **Zip Generator**
```python
# Zip multiple data sources
def multi_source_processor(source1, source2, source3):
    """Process data from multiple sources simultaneously"""
    for data1, data2, data3 in zip(source1, source2, source3):
        combined_record = {
            'sensor_data': data1,
            'weather_data': data2,
            'location_data': data3,
            'correlation_score': calculate_correlation(data1, data2, data3)
        }
        yield combined_record

# Process multiple streams efficiently
sensor_stream = sensor_data_generator()
weather_stream = weather_data_generator()
location_stream = location_data_generator()

for combined in multi_source_processor(sensor_stream, weather_stream, location_stream):
    store_correlated_data(combined)
```

---

## 🔄 4. Coroutines (Bidirectional Generators)

### **Basic Coroutine**
```python
def data_accumulator():
    """Coroutine that accumulates data and provides statistics"""
    total = 0
    count = 0
    
    while True:
        # Receive data
        value = yield
        if value is not None:
            total += value
            count += 1
        
        # Send back current average
        average = total / count if count > 0 else 0
        result = yield average

# Usage
accumulator = data_accumulator()
next(accumulator)  # Prime the coroutine

# Send data and get results
accumulator.send(10)
avg1 = next(accumulator)  # 10.0

accumulator.send(20)
avg2 = next(accumulator)  # 15.0
```

### **Advanced Coroutine: Real-time Data Processor**
```python
def real_time_anomaly_detector(threshold=2.0):
    """Coroutine for real-time anomaly detection"""
    data_history = []
    
    while True:
        # Receive new data point
        new_value = yield
        
        if new_value is not None:
            data_history.append(new_value)
            
            # Keep only last 100 points for moving statistics
            if len(data_history) > 100:
                data_history.pop(0)
            
            # Calculate z-score for anomaly detection
            if len(data_history) > 10:
                mean = sum(data_history) / len(data_history)
                variance = sum((x - mean) ** 2 for x in data_history) / len(data_history)
                std_dev = variance ** 0.5
                
                if std_dev > 0:
                    z_score = abs(new_value - mean) / std_dev
                    is_anomaly = z_score > threshold
                    
                    result = {
                        'value': new_value,
                        'z_score': z_score,
                        'is_anomaly': is_anomaly,
                        'mean': mean,
                        'std_dev': std_dev
                    }
                    
                    anomaly_result = yield result

# Real-time processing
detector = real_time_anomaly_detector(threshold=2.5)
next(detector)  # Prime

for sensor_reading in live_sensor_feed():
    detector.send(sensor_reading)
    analysis = next(detector)
    
    if analysis and analysis['is_anomaly']:
        trigger_alert(analysis)
```

---

## 🌐 5. Async Generators (Python 3.6+)

### **Async Data Fetcher**
```python
import asyncio
import aiohttp

async def async_data_fetcher(urls):
    """Asynchronously fetch data from multiple URLs"""
    async with aiohttp.ClientSession() as session:
        for url in urls:
            async with session.get(url) as response:
                data = await response.json()
                yield data

# Usage
async def process_async_data():
    urls = ['http://api1.com/data', 'http://api2.com/data']
    
    async for data in async_data_fetcher(urls):
        processed = await process_data_async(data)
        await store_data_async(processed)
```

### **Advanced: Async Stream Processor**
```python
import asyncio
import aiofiles

async def async_log_processor(log_files):
    """Process multiple log files asynchronously"""
    tasks = []
    
    async def process_single_file(filename):
        async with aiofiles.open(filename, 'r') as file:
            async for line in file:
                if 'ERROR' in line:
                    yield {
                        'filename': filename,
                        'line': line.strip(),
                        'timestamp': line[:19]
                    }
    
    # Process all files concurrently
    for filename in log_files:
        async for error_data in process_single_file(filename):
            yield error_data

async def real_time_monitoring():
    """Real-time log monitoring system"""
    log_files = ['app1.log', 'app2.log', 'app3.log', 'app4.log']
    
    async for error in async_log_processor(log_files):
        # Process errors in real-time
        await send_alert_async(error)
        await update_dashboard_async(error)
```

---

## 🌳 6. Recursive Generators

### **Directory Tree Walker**
```python
import os

def walk_directory_tree(path):
    """Recursively walk directory tree using generator"""
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        
        if os.path.isfile(item_path):
            yield ('file', item_path)
        elif os.path.isdir(item_path):
            yield ('dir', item_path)
            # Recursive call
            yield from walk_directory_tree(item_path)

# Find all Python files
def find_python_files(root_directory):
    for item_type, path in walk_directory_tree(root_directory):
        if item_type == 'file' and path.endswith('.py'):
            yield path
```

### **Graph Traversal Generator**
```python
def depth_first_search(graph, start_node, visited=None):
    """DFS traversal using generator"""
    if visited is None:
        visited = set()
    
    visited.add(start_node)
    yield start_node
    
    for neighbor in graph.get(start_node, []):
        if neighbor not in visited:
            yield from depth_first_search(graph, neighbor, visited)

def breadth_first_search(graph, start_node):
    """BFS traversal using generator"""
    from collections import deque
    
    queue = deque([start_node])
    visited = {start_node}
    
    while queue:
        current = queue.popleft()
        yield current
        
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Social network analysis
social_graph = {
    'Alice': ['Bob', 'Charlie'],
    'Bob': ['Alice', 'David'],
    'Charlie': ['Alice', 'Eve'],
    'David': ['Bob'],
    'Eve': ['Charlie']
}

# Find all connections from Alice
for person in depth_first_search(social_graph, 'Alice'):
    analyze_connection(person)
```

---

## 🔗 7. Chained Generators (Pipeline Pattern)

### **Complete Data Pipeline**
```python
def read_data(filename):
    """Stage 1: Read raw data"""
    with open(filename, 'r') as file:
        for line in file:
            yield line.strip()

def parse_data(lines):
    """Stage 2: Parse data format"""
    for line in lines:
        if line and not line.startswith('#'):
            parts = line.split(',')
            if len(parts) >= 3:
                yield {
                    'id': parts[0],
                    'value': float(parts[1]),
                    'timestamp': parts[2]
                }

def validate_data(records):
    """Stage 3: Validate data quality"""
    for record in records:
        if (record['value'] >= 0 and 
            record['id'] and 
            record['timestamp']):
            yield record

# Build complete pipeline
def build_pipeline(filename):
    return validate_data(parse_data(read_data(filename)))

# Process efficiently
for processed_record in build_pipeline('massive_dataset.csv'):
    save_to_warehouse(processed_record)
```

### **Parallel Pipeline Processing**
```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def parallel_pipeline_processor(data_source, num_processes=4):
    """Process data pipeline in parallel"""
    
    def process_chunk(chunk):
        """Process a chunk of data through pipeline"""
        results = []
        for item in chunk:
            # Apply all pipeline stages
            processed = transform_data([enrich_data([validate_data([parse_data([item])])])])
            results.extend(list(processed))
        return results
    
    # Create chunks for parallel processing
    chunk_size = 1000
    chunk = []
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        
        for item in data_source:
            chunk.append(item)
            
            if len(chunk) >= chunk_size:
                # Submit chunk for processing
                future = executor.submit(process_chunk, chunk)
                futures.append(future)
                chunk = []
        
        # Process remaining chunk
        if chunk:
            future = executor.submit(process_chunk, chunk)
            futures.append(future)
        
        # Yield results as they complete
        for future in futures:
            for result in future.result():
                yield result
```

---

## 🎯 8. Specialized Generator Patterns

### **Sliding Window Generator**
```python
def sliding_window(iterable, window_size):
    """Create sliding window over data stream"""
    from collections import deque
    
    window = deque(maxlen=window_size)
    
    for item in iterable:
        window.append(item)
        if len(window) == window_size:
            yield list(window)

# Time series analysis
def analyze_time_series(data_stream, window_size=10):
    for window in sliding_window(data_stream, window_size):
        trend = calculate_trend(window)
        yield {
            'window': window,
            'trend': trend,
            'prediction': predict_next_value(window)
        }
```

### **Batching Generator**
```python
def batch_generator(iterable, batch_size):
    """Group items into batches"""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    if batch:
        yield batch

# Database bulk operations
def bulk_insert_processor(data_stream, batch_size=1000):
    for batch in batch_generator(data_stream, batch_size):
        database.bulk_insert(batch)
        print(f"Processed batch of {len(batch)} records")
```

### **Tee Generator (Multiple Consumers)**
```python
import itertools

def tee_processor(data_stream, num_copies=2):
    """Split data stream for multiple consumers"""
    iterators = itertools.tee(data_stream, num_copies)
    return iterators

def multi_consumer_pipeline(data_source):
    """Process same data through multiple pipelines"""
    
    # Split stream into multiple iterators
    stream1, stream2, stream3 = tee_processor(data_source, 3)
    
    # Different processing for each stream
    def analytics_processor(stream):
        for item in stream:
            yield calculate_analytics(item)
    
    def ml_processor(stream):
        for item in stream:
            yield run_ml_prediction(item)
    
    def reporting_processor(stream):
        for item in stream:
            yield generate_report(item)
    
    # Process streams independently
    analytics_results = analytics_processor(stream1)
    ml_results = ml_processor(stream2)
    reporting_results = reporting_processor(stream3)
    
    return analytics_results, ml_results, reporting_results

# Multi-purpose data processing
data_source = large_data_generator()
analytics, predictions, reports = multi_consumer_pipeline(data_source)

# Consume results independently
for analytic in analytics:
    store_analytics(analytic)

for prediction in predictions:
    store_prediction(prediction)

for report in reports:
    generate_dashboard(report)
```

---

## 📊 Performance Comparison Matrix

| Generator Type | Memory Usage | CPU Overhead | Complexity | Best For |
|----------------|--------------|--------------|------------|----------|
| **Function Generator** | Very Low | Low | Medium | Custom logic |
| **Generator Expression** | Very Low | Very Low | Low | Simple filtering |
| **Built-in Generators** | Very Low | Very Low | Low | Standard patterns |
| **Coroutines** | Low | Medium | High | Bidirectional comm |
| **Async Generators** | Low | Medium | High | I/O operations |
| **Recursive Generators** | Medium | Medium | High | Tree traversal |
| **Chained Generators** | Very Low | Low | Medium | Pipeline processing |

---

## 🚀 Best Practices for Big Data

### **1. Memory Management**
```python
# ✅ Good - Memory efficient
def process_large_dataset():
    for record in read_large_file():
        yield transform_record(record)

# ❌ Bad - Loads everything in memory
def process_large_dataset_bad():
    all_records = list(read_large_file())
    return [transform_record(r) for r in all_records]
```

### **2. Error Handling**
```python
def robust_generator(data_source):
    """Generator with proper error handling"""
    try:
        for item in data_source:
            try:
                processed = process_item(item)
                yield processed
            except ProcessingError as e:
                logger.error(f"Error processing item: {e}")
                continue
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        raise
    finally:
        cleanup_resources()
```

### **3. Progress Monitoring**
```python
def progress_generator(iterable, total=None):
    """Generator with progress monitoring"""
    count = 0
    for item in iterable:
        count += 1
        if count % 1000 == 0:
            if total:
                progress = (count / total) * 100
                print(f"Progress: {progress:.1f}% ({count}/{total})")
            else:
                print(f"Processed: {count} items")
        yield item
```

---

## 🔥 Generator Anti-Patterns to Avoid

### **1. Converting to List Unnecessarily**
```python
# ❌ Bad
gen = (x for x in huge_dataset)
all_items = list(gen)  # Defeats the purpose

# ✅ Good
gen = (x for x in huge_dataset)
for item in gen:
    process_item(item)
```

### **2. Multiple Iterations**
```python
# ❌ Bad - Generator exhausted
gen = (x**2 for x in range(1000))
sum1 = sum(gen)  # Works
sum2 = sum(gen)  # Returns 0

# ✅ Good - Create generator function
def squares_generator():
    return (x**2 for x in range(1000))

sum1 = sum(squares_generator())
sum2 = sum(squares_generator())
```

### **3. Premature Optimization**
```python
# ❌ Bad - Unnecessary complexity for small datasets
def small_data_generator():
    for i in range(10):  # Small dataset
        yield complex_processing(i)

# ✅ Good - Use list for small datasets
def small_data_processor():
    return [complex_processing(i) for i in range(10)]
```

---

## 📚 Integration with Big Data Tools

### **Apache Spark Integration**
```python
from pyspark.sql import SparkSession

def spark_generator_integration():
    spark = SparkSession.builder.appName("GeneratorIntegration").getOrCreate()
    
    def data_generator(num_records):
        for i in range(num_records):
            yield {
                'id': i,
                'value': random.random(),
                'category': random.choice(['A', 'B', 'C'])
            }
    
    # Convert generator to Spark DataFrame
    data_gen = data_generator(1000000)
    df = spark.createDataFrame(data_gen)
    
    result = df.groupBy('category').avg('value').collect()
    return result
```

### **Pandas Integration**
```python
import pandas as pd

def pandas_chunk_processor(filename, chunk_size=10000):
    """Process large files with pandas and generators"""
    
    def chunk_generator():
        for chunk in pd.read_csv(filename, chunksize=chunk_size):
            # Process each chunk
            processed_chunk = chunk.groupby('category').sum()
            yield processed_chunk
    
    # Combine all chunks
    combined_result = pd.concat(chunk_generator(), ignore_index=True)
    return combined_result
```

---

## 🎯 Summary and Recommendations

### **When to Use Each Generator Type:**

1. **Function Generators**: Complex business logic, stateful processing  
2. **Generator Expressions**: Simple transformations, filtering operations  
3. **Built-in Generators**: Standard iteration patterns  
4. **Coroutines**: Bidirectional data flow, real-time processing  
5. **Async Generators**: I/O bound operations, concurrent processing  
6. **Recursive Generators**: Tree/graph traversal, hierarchical data  
7. **Chained Generators**: Multi-stage data pipelines  

### **Performance Tips:**
- Use generators for large datasets (>1GB)
- Chain generators for complex pipelines  
- Avoid converting generators to lists unless necessary  
- Use `itertools` for advanced generator operations  
- Monitor memory usage with `tracemalloc`  

### **Big Data Scenarios:**
- **ETL Pipelines**: Chained generators for multi-stage processing  
- **Real-time Analytics**: Coroutines and async generators  
- **Log Processing**: Function generators with error handling  
- **Data Quality**: Generator expressions for filtering  
- **Machine Learning**: Generators for batch processing training data  

---

> 💡 **Pro Tip**: The key to effective generator usage is understanding your data size, processing requirements, and memory constraints. Choose the right generator type based on your specific use case! 