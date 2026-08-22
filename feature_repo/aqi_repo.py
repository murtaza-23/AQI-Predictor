from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

# A single-location project doesn't need a real entity key
# but Feast requires one (a constant "location" entity)
location = Entity(name="location", join_keys=["location_id"])

aqi_source = FileSource(
    name="aqi_features_source",
    path="../data/aqi_features.parquet",
    timestamp_field="timestamp",
)

aqi_feature_view = FeatureView(
    name="aqi_features",
    entities=[location],
    ttl=timedelta(days=400),
    schema=[
        Field(name="aqi", dtype=Float64),
        Field(name="pm2_5", dtype=Float64),
        Field(name="pm10", dtype=Float64),
        Field(name="o3", dtype=Float64),
        Field(name="no2", dtype=Float64),
        Field(name="co", dtype=Float64),
        Field(name="so2", dtype=Float64),
        Field(name="hour", dtype=Int64),
        Field(name="day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="hour_category", dtype=Int64),
    ],
    source=aqi_source,
    online=True,
)