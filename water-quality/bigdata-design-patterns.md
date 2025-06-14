# Big Data Design Patterns: Scenario-Based System Design

## Complete Guide to HLD and LLD for Big Data Systems

This guide covers essential big data design patterns through water quality monitoring scenarios, with both High-Level Design (HLD) and Low-Level Design (LLD) considerations.

---

## **Design Pattern 1: Lambda Architecture**

### **Scenario: European Water Quality Compliance System**
**Context:** EU requires both real-time pollution alerts AND accurate historical compliance reports from 27 countries.

### **HLD Questions:**
1. How do you design a system that provides both real-time and batch processing?
2. What are the main architectural components needed?
3. How do you handle data consistency between real-time and batch layers?
4. What's your strategy for handling late-arriving data?

### **LLD Questions:**
1. Which specific technologies will you use for each layer?
2. How do you implement data reconciliation between speed and batch layers?
3. What's your schema design for the serving layer?
4. How do you handle version control and deployment across layers?

### **Lambda Architecture Components:**

```
┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   New Data      │
│ (Sensors, APIs) │────│   All Data      │
└─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Message Queue  │
                    │    (Kafka)      │
                    └─────────────────┘
                           │     │
                           ▼     ▼
                  ┌─────────┐  ┌─────────┐
                  │ Speed   │  │ Batch   │
                  │ Layer   │  │ Layer   │
                  │(Storm)  │  │(Spark)  │
                  └─────────┘  └─────────┘
                      │             │
                      ▼             ▼
                  ┌─────────┐  ┌─────────┐
                  │Real-time│  │ Master  │
                  │  View   │  │Dataset  │
                  │(Cassandra)│ │(HDFS)  │
                  └─────────┘  └─────────┘
                      │             │
                      └─────┬───────┘
                            ▼
                    ┌─────────────────┐
                    │  Serving Layer  │
                    │   (Combined)    │
                    └─────────────────┘
```

### **Implementation Details:**

**HLD Components:**
- **Speed Layer:** Apache Storm/Spark Streaming for real-time processing
- **Batch Layer:** Apache Spark/Hadoop for comprehensive processing
- **Serving Layer:** Combined views for querying

**LLD Implementation:**
```python
# Speed Layer - Real-time Processing
class SpeedLayer:
    def __init__(self):
        self.kafka_consumer = KafkaConsumer('water-quality-topic')
        self.cassandra_client = CassandraClient()
    
    def process_real_time(self, data):
        # Real-time aggregation
        result = self.calculate_immediate_stats(data)
        self.cassandra_client.insert_real_time_view(result)

# Batch Layer - Historical Processing
class BatchLayer:
    def __init__(self):
        self.spark_session = SparkSession.builder.appName("WaterQuality").getOrCreate()
        self.hdfs_client = HDFSClient()
    
    def process_batch(self):
        # Read all historical data
        df = self.spark_session.read.format("parquet").load("hdfs://water-quality/")
        # Comprehensive calculations
        batch_views = self.calculate_comprehensive_stats(df)
        self.hdfs_client.save_batch_views(batch_views)
```

**When to Use Lambda:**
- ✅ Need both real-time and accurate historical analytics
- ✅ Can tolerate some complexity for completeness
- ✅ Have resources to maintain two processing systems

---

## **Design Pattern 2: Kappa Architecture**

### **Scenario: Real-Time Water Contamination Detection**
**Context:** City needs immediate contamination alerts with ability to reprocess historical data when detection algorithms improve.

### **HLD Questions:**
1. How do you design a stream-only architecture?
2. What's your strategy for reprocessing historical data?
3. How do you handle different processing speeds for different use cases?
4. What's your approach to data versioning and schema evolution?

### **LLD Questions:**
1. How do you implement replay functionality in your streaming system?
2. What's your partitioning strategy for Kafka topics?
3. How do you handle state management in stream processing?
4. What's your approach to exactly-once processing guarantees?

### **Kappa Architecture Flow:**

```
┌─────────────────┐
│   Data Sources  │
│ (Sensors, APIs) │
└─────────────────┘
          │
          ▼
┌─────────────────┐
│  Kafka Topics   │
│  (Partitioned)  │
└─────────────────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐
│ Stream Process 1│    │ Stream Process 2│
│(Real-time Alert)│    │(Hourly Reports) │
└─────────────────┘    └─────────────────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│   Fast Storage  │    │   Batch Storage │
│   (Cassandra)   │    │     (HDFS)      │
└─────────────────┘    └─────────────────┘
```

### **Implementation Details:**

**HLD Components:**
- **Single Stream:** All data flows through Kafka
- **Multiple Consumers:** Different processing speeds for different needs
- **Replay Capability:** Reprocess from any point in time

**LLD Implementation:**
```scala
// Stream Processing with Kafka Streams
class WaterQualityProcessor {
  val streams = new KafkaStreams(buildTopology(), props)
  
  def buildTopology(): Topology = {
    val builder = new StreamsBuilder()
    
    // Real-time contamination detection
    val readings = builder.stream[String, WaterQualityReading]("sensor-data")
    
    readings
      .filter((_, reading) => reading.isContaminated())
      .to("contamination-alerts")
    
    // Hourly aggregations
    readings
      .groupByKey()
      .windowedBy(TimeWindows.of(Duration.ofHours(1)))
      .aggregate(
        () => new HourlyStats(),
        (_, reading, stats) => stats.update(reading),
        Materialized.as("hourly-stats")
      )
      .toStream()
      .to("hourly-reports")
    
    builder.build()
  }
}

// Replay mechanism for reprocessing
class ReplayService {
  def reprocessFromTimestamp(timestamp: Long): Unit = {
    val consumer = new KafkaConsumer[String, WaterQualityReading](props)
    consumer.seek(partition, timestamp)
    // Reprocess with updated algorithms
  }
}
```

**When to Use Kappa:**
- ✅ Primarily need real-time processing
- ✅ Want to avoid complexity of dual systems
- ✅ Need ability to reprocess historical data with new algorithms

---

## **Design Pattern 3: Data Lake Architecture**

### **Scenario: Multi-Source Environmental Data Integration**
**Context:** Combine sensor data, satellite imagery, weather APIs, social media, and government reports in their raw formats.

### **HLD Questions:**
1. How do you design storage for multiple data formats and sources?
2. What's your data governance and catalog strategy?
3. How do you handle data quality and lineage tracking?
4. What's your approach to data security and access control?

### **LLD Questions:**
1. What's your folder structure and partitioning strategy?
2. How do you implement schema-on-read vs schema-on-write?
3. What file formats will you use for different data types?
4. How do you implement data lifecycle management?

### **Data Lake Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│                    Data Sources                         │
├─────────────┬─────────────┬─────────────┬─────────────┤
│   Sensors   │  Satellite  │   Weather   │Social Media │
│    APIs     │   Imagery   │    APIs     │    Feeds    │
└─────────────┴─────────────┴─────────────┴─────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  Ingestion Layer                        │
├─────────────┬─────────────┬─────────────┬─────────────┤
│Apache NiFi  │   Kafka     │  Custom     │   REST      │
│  Flows      │ Connectors  │ Scrapers    │   APIs      │
└─────────────┴─────────────┴─────────────┴─────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Data Lake Storage                    │
│                        (S3/HDFS)                       │
├─────────────┬─────────────┬─────────────┬─────────────┤
│    Raw      │ Processed   │  Curated    │  Archived   │
│   Zone      │    Zone     │    Zone     │    Zone     │
└─────────────┴─────────────┴─────────────┴─────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                Processing & Analytics                   │
├─────────────┬─────────────┬─────────────┬─────────────┤
│Apache Spark │  Apache     │   Machine   │   Query     │
│   Batch     │  Streaming  │  Learning   │  Engines    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### **Implementation Details:**

**HLD Components:**
- **Raw Zone:** Store data in original format
- **Processed Zone:** Cleaned and standardized data
- **Curated Zone:** Business-ready datasets
- **Metadata Catalog:** Track schemas, lineage, quality

**LLD Implementation:**
```python
# Data Lake Folder Structure
"""
/water-quality-lake/
├── raw/
│   ├── year=2024/month=01/day=15/
│   │   ├── sensors/
│   │   │   └── sensor_data_20240115.json
│   │   ├── satellite/
│   │   │   └── imagery_20240115.tiff
│   │   └── weather/
│   │       └── weather_20240115.xml
├── processed/
│   ├── sensors/
│   │   └── year=2024/month=01/day=15/
│   │       └── cleaned_sensor_data.parquet
└── curated/
    ├── daily_reports/
    └── compliance_metrics/
"""

# Data Processing Pipeline
class DataLakeProcessor:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.catalog = DataCatalog()
    
    def process_raw_to_processed(self, source_path: str):
        # Read raw data with schema inference
        raw_df = self.spark.read.option("inferSchema", "true").json(source_path)
        
        # Clean and standardize
        cleaned_df = self.clean_data(raw_df)
        
        # Write as Parquet with partitioning
        cleaned_df.write \
            .partitionBy("year", "month", "day") \
            .mode("overwrite") \
            .parquet("/processed/sensors/")
        
        # Update catalog
        self.catalog.register_dataset("processed_sensors", cleaned_df.schema)
    
    def create_curated_view(self):
        # Join multiple processed datasets
        sensors = self.spark.read.parquet("/processed/sensors/")
        weather = self.spark.read.parquet("/processed/weather/")
        
        # Create business-ready dataset
        curated = sensors.join(weather, ["date", "location"])
        
        # Write to curated zone
        curated.write.mode("overwrite").parquet("/curated/daily_reports/")

# Data Catalog Implementation
class DataCatalog:
    def __init__(self):
        self.metadata_store = CassandraClient()
    
    def register_dataset(self, name: str, schema: StructType):
        metadata = {
            'name': name,
            'schema': schema.json(),
            'created_at': datetime.now(),
            'location': f"/processed/{name}/",
            'format': 'parquet'
        }
        self.metadata_store.insert('data_catalog', metadata)
```

**When to Use Data Lake:**
- ✅ Multiple diverse data sources and formats
- ✅ Exploratory analytics and data science workflows
- ✅ Need to preserve raw data for future processing
- ✅ Schema evolution and flexibility requirements

---

## **Design Pattern 4: Event Sourcing Pattern**

### **Scenario: Water Quality Audit Trail System**
**Context:** Regulatory compliance requires complete audit trail of all sensor readings, calibrations, and system changes.

### **HLD Questions:**
1. How do you design a system where events are the source of truth?
2. What's your strategy for event versioning and schema evolution?
3. How do you handle event replay and system recovery?
4. What's your approach to snapshotting for performance?

### **LLD Questions:**
1. How do you implement event store with high write throughput?
2. What's your event serialization and compression strategy?
3. How do you implement eventual consistency across read models?
4. What's your approach to handling event ordering and timestamps?

### **Event Sourcing Architecture:**

```
┌─────────────────┐    ┌─────────────────┐
│   Commands      │    │     Events      │
│ (Calibrate,     │───▶│ (Calibrated,    │
│  ReadSensor)    │    │  SensorRead)    │
└─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Event Store   │
                    │   (Immutable    │
                    │    Append-Only) │
                    └─────────────────┘
                              │
                              ▼
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Current State│    │Audit Trail  │    │Compliance   │
│ Read Model  │    │ Read Model  │    │ Read Model  │
│(Cassandra)  │    │(Elasticsearch)│   │(PostgreSQL)│
└─────────────┘    └─────────────┘    └─────────────┘
```

### **Implementation Details:**

**HLD Components:**
- **Event Store:** Immutable log of all events
- **Command Handlers:** Process business commands
- **Event Handlers:** Update read models
- **Read Models:** Optimized views for different use cases

**LLD Implementation:**
```python
# Event Store Implementation
class WaterQualityEvent:
    def __init__(self, event_type: str, aggregate_id: str, data: dict):
        self.event_id = uuid.uuid4()
        self.event_type = event_type
        self.aggregate_id = aggregate_id
        self.data = data
        self.timestamp = datetime.utcnow()
        self.version = 1

class EventStore:
    def __init__(self):
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode()
        )
    
    def append_event(self, event: WaterQualityEvent):
        event_data = {
            'event_id': str(event.event_id),
            'event_type': event.event_type,
            'aggregate_id': event.aggregate_id,
            'data': event.data,
            'timestamp': event.timestamp.isoformat(),
            'version': event.version
        }
        
        # Append to event stream
        self.kafka_producer.send(
            topic=f"water-quality-events",
            key=event.aggregate_id,
            value=event_data
        )
    
    def get_events(self, aggregate_id: str, from_version: int = 0):
        # Read events from Kafka or persistent store
        consumer = KafkaConsumer(
            'water-quality-events',
            auto_offset_reset='earliest'
        )
        
        events = []
        for message in consumer:
            event_data = json.loads(message.value)
            if (event_data['aggregate_id'] == aggregate_id and 
                event_data['version'] >= from_version):
                events.append(event_data)
        
        return events

# Aggregate Root
class WaterQualitySensor:
    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self.events = []
        self.version = 0
        self.last_reading = None
        self.calibration_date = None
    
    def record_reading(self, ph: float, temperature: float):
        event = WaterQualityEvent(
            'SensorReadingRecorded',
            self.sensor_id,
            {
                'ph': ph,
                'temperature': temperature,
                'reading_time': datetime.utcnow().isoformat()
            }
        )
        self.apply_event(event)
        return event
    
    def calibrate_sensor(self, calibration_data: dict):
        event = WaterQualityEvent(
            'SensorCalibrated',
            self.sensor_id,
            calibration_data
        )
        self.apply_event(event)
        return event
    
    def apply_event(self, event: WaterQualityEvent):
        if event.event_type == 'SensorReadingRecorded':
            self.last_reading = event.data
        elif event.event_type == 'SensorCalibrated':
            self.calibration_date = event.timestamp
        
        self.events.append(event)
        self.version += 1
    
    def replay_events(self, events: list):
        for event_data in events:
            event = WaterQualityEvent(
                event_data['event_type'],
                event_data['aggregate_id'],
                event_data['data']
            )
            self.apply_event(event)

# Read Model Projector
class ComplianceReportProjector:
    def __init__(self):
        self.postgres_client = PostgreSQLClient()
    
    def handle_sensor_reading_recorded(self, event: WaterQualityEvent):
        # Update compliance read model
        self.postgres_client.execute("""
            INSERT INTO compliance_readings 
            (sensor_id, ph, temperature, recorded_at)
            VALUES (%s, %s, %s, %s)
        """, [
            event.aggregate_id,
            event.data['ph'],
            event.data['temperature'],
            event.data['reading_time']
        ])
    
    def handle_sensor_calibrated(self, event: WaterQualityEvent):
        # Update calibration tracking
        self.postgres_client.execute("""
            INSERT INTO sensor_calibrations
            (sensor_id, calibrated_at, calibration_data)
            VALUES (%s, %s, %s)
        """, [
            event.aggregate_id,
            event.timestamp,
            json.dumps(event.data)
        ])
```

**When to Use Event Sourcing:**
- ✅ Need complete audit trail and compliance
- ✅ Complex business logic with state changes
- ✅ Require ability to replay and reconstruct state
- ✅ Multiple read models from same data

---

## **Design Pattern 5: CQRS (Command Query Responsibility Segregation)**

### **Scenario: High-Performance Water Quality Dashboard**
**Context:** Public dashboard serves millions of users while operators perform thousands of sensor updates per minute.

### **HLD Questions:**
1. How do you separate read and write concerns for optimal performance?
2. What's your strategy for eventual consistency between read and write models?
3. How do you handle complex queries without impacting write performance?
4. What's your approach to scaling reads and writes independently?

### **LLD Questions:**
1. How do you implement command validation and business rules?
2. What's your strategy for read model synchronization?
3. How do you handle read model failures and recovery?
4. What's your caching strategy for read models?

### **CQRS Architecture:**

```
┌─────────────────┐    ┌─────────────────┐
│   Commands      │    │    Queries      │
│(Update Sensor,  │    │(Get Dashboard,  │
│ Add Alert)      │    │ Generate Report)│
└─────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│  Command Side   │    │   Query Side    │
│                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Command      │ │    │ │Query        │ │
│ │Handlers     │ │    │ │Handlers     │ │
│ └─────────────┘ │    │ └─────────────┘ │
│         │       │    │         │       │
│         ▼       │    │         ▼       │
│ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Write        │ │    │ │Read         │ │
│ │Database     │ │    │ │Database     │ │
│ │(PostgreSQL) │ │    │ │(Elasticsearch)│ │
│ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘
         │                        ▲
         │       Events            │
         └────────────────────────┘
```

### **Implementation Details:**

**LLD Implementation:**
```python
# Command Side
class SensorUpdateCommand:
    def __init__(self, sensor_id: str, reading_data: dict):
        self.sensor_id = sensor_id
        self.reading_data = reading_data
        self.command_id = uuid.uuid4()
        self.timestamp = datetime.utcnow()

class SensorCommandHandler:
    def __init__(self):
        self.write_db = PostgreSQLClient()
        self.event_publisher = EventPublisher()
    
    def handle_sensor_update(self, command: SensorUpdateCommand):
        # Validate command
        if not self.validate_sensor_data(command.reading_data):
            raise ValidationError("Invalid sensor data")
        
        # Write to command database
        self.write_db.execute("""
            INSERT INTO sensor_readings 
            (sensor_id, ph, temperature, timestamp)
            VALUES (%s, %s, %s, %s)
        """, [
            command.sensor_id,
            command.reading_data['ph'],
            command.reading_data['temperature'],
            command.timestamp
        ])
        
        # Publish event for read side
        event = SensorReadingUpdatedEvent(
            command.sensor_id,
            command.reading_data,
            command.timestamp
        )
        self.event_publisher.publish(event)
    
    def validate_sensor_data(self, data: dict) -> bool:
        return (0 <= data.get('ph', 0) <= 14 and 
                -10 <= data.get('temperature', 0) <= 50)

# Query Side
class DashboardQuery:
    def __init__(self, location: str, time_range: tuple):
        self.location = location
        self.time_range = time_range

class DashboardQueryHandler:
    def __init__(self):
        self.read_db = ElasticsearchClient()
        self.cache = RedisClient()
    
    def handle_dashboard_query(self, query: DashboardQuery):
        # Check cache first
        cache_key = f"dashboard:{query.location}:{query.time_range}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Query read database
        result = self.read_db.search(
            index='water-quality-dashboard',
            body={
                'query': {
                    'bool': {
                        'must': [
                            {'term': {'location': query.location}},
                            {'range': {
                                'timestamp': {
                                    'gte': query.time_range[0],
                                    'lte': query.time_range[1]
                                }
                            }}
                        ]
                    }
                }
            }
        )
        
        # Cache result
        self.cache.setex(cache_key, 300, json.dumps(result))
        return result

# Read Model Synchronizer
class ReadModelSynchronizer:
    def __init__(self):
        self.elasticsearch = ElasticsearchClient()
        self.event_consumer = KafkaConsumer('sensor-events')
    
    def start_sync(self):
        for message in self.event_consumer:
            event = json.loads(message.value)
            self.handle_event(event)
    
    def handle_event(self, event: dict):
        if event['type'] == 'SensorReadingUpdated':
            self.update_dashboard_model(event)
            self.update_analytics_model(event)
    
    def update_dashboard_model(self, event: dict):
        doc = {
            'sensor_id': event['sensor_id'],
            'location': event['location'],
            'ph': event['data']['ph'],
            'temperature': event['data']['temperature'],
            'timestamp': event['timestamp'],
            'status': self.calculate_status(event['data'])
        }
        
        self.elasticsearch.index(
            index='water-quality-dashboard',
            id=f"{event['sensor_id']}_{event['timestamp']}",
            body=doc
        )
    
    def calculate_status(self, data: dict) -> str:
        ph = data['ph']
        if 6.5 <= ph <= 8.5:
            return 'GOOD'
        elif 6.0 <= ph <= 9.0:
            return 'ACCEPTABLE'
        else:
            return 'POOR'
```

**When to Use CQRS:**
- ✅ Different performance requirements for reads vs writes
- ✅ Complex queries that would slow down operational database
- ✅ Need to scale read and write operations independently
- ✅ Different security models for commands vs queries

---

## **HLD vs LLD Comparison Matrix**

| Aspect | High-Level Design (HLD) | Low-Level Design (LLD) |
|--------|-------------------------|------------------------|
| **Focus** | System architecture, components | Implementation details, code structure |
| **Audience** | Architects, stakeholders | Developers, implementers |
| **Scope** | Overall system design | Module/component design |
| **Technologies** | Technology categories | Specific tools and frameworks |
| **Documentation** | Architecture diagrams, flow charts | Class diagrams, APIs, schemas |

### **HLD Questions Template:**
1. What are the main system components?
2. How do components communicate?
3. What are the data flow patterns?
4. How do you handle scalability and reliability?
5. What are the security and compliance requirements?

### **LLD Questions Template:**
1. What specific technologies and frameworks?
2. What are the data models and schemas?
3. How do you implement error handling?
4. What are the API contracts?
5. How do you handle deployment and monitoring?

---

## **Pattern Selection Decision Tree**

```
Start: What's your primary requirement?

├── Real-time + Historical Accuracy
│   └── Lambda Architecture
│
├── Primarily Real-time with Replay Capability
│   └── Kappa Architecture
│
├── Multiple Data Sources/Formats
│   └── Data Lake Architecture
│
├── Complete Audit Trail Required
│   └── Event Sourcing
│
├── High Read/Write Performance Needs
│   └── CQRS
│
└── Simple Analytics Requirements
    └── Traditional Data Warehouse
```

## **Best Practices Summary**

### **Architecture Design:**
1. **Start Simple:** Begin with traditional patterns, evolve to big data patterns as needed
2. **Identify Bottlenecks:** Understand your specific performance and scale requirements
3. **Plan for Evolution:** Design systems that can migrate between patterns
4. **Consider Team Skills:** Choose patterns your team can implement and maintain

### **Implementation Guidelines:**
1. **Prototype First:** Build small proofs of concept before full implementation
2. **Monitor Everything:** Implement comprehensive monitoring from day one
3. **Document Decisions:** Record why you chose specific patterns and technologies
4. **Plan for Failure:** Design error handling and recovery mechanisms

### **Technology Selection:**
1. **Proven Solutions:** Choose mature technologies for production systems
2. **Community Support:** Consider ecosystem and community around technologies
3. **Vendor Lock-in:** Evaluate dependency on specific cloud providers or vendors
4. **Total Cost:** Consider licensing, infrastructure, and operational costs

**Remember:** The best architecture is the one that solves your specific problems with the least complexity. Start with your requirements, not with the pattern! 