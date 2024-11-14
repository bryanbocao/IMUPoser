# create some polars data
import polars
data = [(1.0, 1.0, 1.0, 7.0, 1.5, -2.3), 
        (2.0, None, 2.0, 7.0, 8.5, 6.7), 
        (2.0, None, 3.0, 7.0, -2.3, 4.4),
        (3.0, 4.0, 3.0, 7.0, 0.0, 0.0),
        (4.0, 5.0, 4.0, 7.0, 12.1, -5.2),
        (1.0, 1.0, 1.0, 7.0, 1.5, -2.3), 
        (2.0, None, 2.0, 7.0, 8.5, 6.7), 
        (2.0, None, 3.0, 7.0, -2.3, 4.4),
        (3.0, 4.0, 3.0, 7.0, 0.0, 0.0),
        (4.0, 5.0, 3.9, 7.0, 12.1, -5.2),
        (1.0, 1.0, 1.0, 7.0, 1.5, -2.3), 
        (2.0, None, 2.0, 7.0, 8.5, 6.7), 
        (2.0, None, 3.0, 7.0, -2.3, 4.4),
        (3.0, 4.0, 3.0, 7.0, 0.0, 0.0),
        (4.0, 5.0, 4.1, 7.0, 12.1, -5.2),
        ]

columns = ["target", "some_null", "feature", "constant", "other_feature", "another_feature"]
df_polars = polars.DataFrame(data=data, schema=columns)

# select top 2 features using mRMR
import mrmr
selected_features = mrmr.polars.mrmr_regression(df=df_polars, target_column="target", K=2, return_scores=True)
print('\nselected_features: ', selected_features)