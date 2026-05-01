-- Initialize the featurestore database schema

CREATE TABLE IF NOT EXISTS trips (
    trip_id VARCHAR(64) PRIMARY KEY,
    request_id VARCHAR(64),
    driver_id VARCHAR(64) NOT NULL,
    rider_id VARCHAR(64) NOT NULL,
    accepted BOOLEAN DEFAULT FALSE,
    completed BOOLEAN DEFAULT FALSE,
    cancelled_by_driver BOOLEAN DEFAULT FALSE,
    cancelled_by_rider BOOLEAN DEFAULT FALSE,
    rating FLOAT,
    rider_rating_given FLOAT,
    tip_pct FLOAT DEFAULT 0.0,
    pickup_lat FLOAT,
    pickup_lng FLOAT,
    destination_lat FLOAT,
    destination_lng FLOAT,
    eta_seconds FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_trips_driver ON trips(driver_id);
CREATE INDEX idx_trips_rider ON trips(rider_id);
CREATE INDEX idx_trips_created ON trips(created_at);
CREATE INDEX idx_trips_request ON trips(request_id);

CREATE TABLE IF NOT EXISTS driver_features (
    driver_id VARCHAR(64) PRIMARY KEY,
    driver_acceptance_rate_7d FLOAT,
    driver_avg_rating_30d FLOAT,
    driver_cancel_rate_7d FLOAT,
    driver_trips_lifetime INT,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rider_features (
    rider_id VARCHAR(64) PRIMARY KEY,
    rider_avg_rating_given FLOAT,
    rider_cancel_rate_30d FLOAT,
    rider_avg_tip_pct FLOAT,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pair_features (
    rider_id VARCHAR(64),
    driver_id VARCHAR(64),
    pair_trip_count INT DEFAULT 0,
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (rider_id, driver_id)
);
