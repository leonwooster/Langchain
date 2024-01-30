```mermaid
flowchart LR
    subgraph DC1 [Data Center 1]
        LB1[Load Balancer 1]
        APP1_1[Application Server 1.1]
        APP1_2[Application Server 1.2]
        CACHE1[Cache Layer 1]
        DB1_1[Database 1 Primary]
        DB1_2[Database 1 Replica]
        LB1 --> APP1_1
        LB1 --> APP1_2
        APP1_1 --> CACHE1
        APP1_2 --> CACHE1
        CACHE1 --> DB1_1
        DB1_1 --> DB1_2
    end
    subgraph DC2 [Data Center 2]
        LB2[Load Balancer 2]
        APP2_1[Application Server 2.1]
        APP2_2[Application Server 2.2]
        CACHE2[Cache Layer 2]
        DB2_1[Database 2 Primary]
        DB2_2[Database 2 Replica]
        LB2 --> APP2_1
        LB2 --> APP2_2
        APP2_1 --> CACHE2
        APP2_2 --> CACHE2
        CACHE2 --> DB2_1
        DB2_1 --> DB2_2
    end
    DB1_1 -.->|Async Replication| DB2_1
    DB2_1 -.->|Async Replication| DB1_1

    classDef center fill:#f9f,stroke:#333,stroke-width:4px;
    class DC1,DC2 center;
