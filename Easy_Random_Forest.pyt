# -*- coding: utf-8 -*-

"""
 _____
|  ___|
| |__  __ _ ___ _   _
|  __|/ _` / __| | | |
| |__| (_| \__ \ |_| |
\____/\__,_|___/\__, |
                 __/ |
                |___/
______                _
| ___ \              | |
| |_/ /__ _ _ __   __| | ___  _ __ ___
|    // _` | '_ \ / _` |/ _ \| '_ ` _ \
| |\ \ (_| | | | | (_| | (_) | | | | | |
\_| \_\__,_|_| |_|\__,_|\___/|_| |_| |_|


______                  _
|  ___|                | |
| |_ ___  _ __ ___  ___| |_
|  _/ _ \| '__/ _ \/ __| __|
| || (_) | | |  __/\__ \ |_
\_| \___/|_|  \___||___/\__|


Even an archaeologist can do it!

Michael Troyer
michael.troyer@usda.gov
5/27/2020

Purpose:

Credits:

Todo:


"""


import datetime
import math
import re
import traceback
import pandas as pd
import numpy as np
import arcpy
import random_forest



def get_acres(fc):
    """Check for an acres field in fc - create if doesn't exist or flag for calculation.
       Recalculate acres and return name of acre field"""

    # Add ACRES field to analysis area - check if exists
    field_list = [field.name for field in arcpy.ListFields(fc, wild_card='Acres')]

    # If *Acres* exists in table, flag for calculation instead
    if field_list:
        acre_field = field_list[0] # select the 'Acres' variant
    else:
        acre_field = "ACRES"
        arcpy.AddField_management(fc, acre_field, "DOUBLE", 15, 2)
    arcpy.CalculateField_management(fc, acre_field, "!shape.area@ACRES!", "PYTHON_9.3")
    acres_sum = sum([row[0] for row in arcpy.da.SearchCursor(fc, acre_field)])
    return {'Acres_Field': acre_field, 'Acres_Sum': acres_sum}


def build_where_clause(table, field, valueList):
    """Takes a list of values and constructs a SQL WHERE
    clause to select those values within a given field and table."""
    fieldDelimited = arcpy.AddFieldDelimiters(arcpy.Describe(table).path, field)
    fieldType = arcpy.ListFields(table, field)[0].type
    if str(fieldType) == 'String':
        valueList = ["'%s'" % value for value in valueList]
    whereClause = "%s IN(%s)" % (fieldDelimited, ', '.join(map(str, valueList)))
    return whereClause


def feature_class_to_data_frame(in_table, filter_fields=None):
    """
    Convert all or a subset of fields in an arcgis table into a pandas dataframe with an object ID index.
    """
    OIDFieldName = arcpy.Describe(in_table).OIDFieldName
    if filter_fields:
        final_fields = [OIDFieldName] + filter_fields
    else:
        final_fields = [field.name for field in arcpy.ListFields(in_table)]
    data = [row for row in arcpy.da.SearchCursor(in_table, final_fields)]
    fc_dataframe = pd.DataFrame(data, columns=final_fields)
    fc_dataframe = fc_dataframe.set_index(OIDFieldName, drop=True)
    return fc_dataframe


def one_hot_dataframe(dataframe, one_hot_columns):
    one_hot_dfs = []
    for col in one_hot_columns:
        one_hot_dfs.append(pd.get_dummies(dataframe[col], prefix=str(col)))
    dataframe.drop(columns=one_hot_columns, inplace=True)
    return dataframe.join(one_hot_dfs)


class InsufficientSurveyCoverage (BaseException): pass
class InsufficientSiteSample     (BaseException): pass


class Toolbox(object):
    def __init__(self):
        self.label = "Easy Random Forest Toolbox"
        self.alias = "Easy_Random_Forest_Toolbox"
        self.tools = [Easy_Random_Forest]


class Easy_Random_Forest(object):
    def __init__(self):
        self.label = "Easy Random Forest"
        self.description = ""
        self.canRunInBackground = True

    def getParameterInfo(self):

        # Input Analysis Area
        input_analysis_area=arcpy.Parameter(
            displayName="Input Modelling Analysis Area",
            name="Input_Analysis_Area",
            datatype="DEFeatureClass",
            )

        # Input Surveys feature class
        surveys=arcpy.Parameter(
            displayName="Existing Surveys Feature Class",
            name="Input_Surveys",
            datatype="DEFeatureClass",
            )

        # Surveys Subselection boolean
        surveys_filter=arcpy.Parameter(
            displayName="Filter Surveys",
            name="Filter_Surveys",
            datatype="Boolean",
            parameterType="Optional",
            enabled = "False",
            )

        # Surveys Subselection field
        surveys_filter_field=arcpy.Parameter(
            displayName="Surveys Filter Field",
            name="Surveys_Filter_Field",
            datatype="String",
            parameterType="Optional",
            enabled = "False",
            )

        # Surveys Subselection value
        surveys_filter_value=arcpy.Parameter(
            displayName="Surveys Filter Value",
            name="Surveys_Filter_Value",
            datatype="String",
            parameterType="Optional",
            )

        # Input Sites feature class
        sites=arcpy.Parameter(
            displayName="Existing Sites Feature Class",
            name="Input_Sites",
            datatype="DEFeatureClass",
            )

        # Sites Subselection boolean
        sites_filter=arcpy.Parameter(
            displayName="Filter Sites",
            name="Filter_Sites",
            datatype="Boolean",
            parameterType="Optional",
            enabled = "False",
            )

        # Sites Subselection field
        sites_filter_field=arcpy.Parameter(
            displayName="Sites Filter Field",
            name="Sites_Filter_Field",
            datatype="String",
            parameterType="Optional",
            enabled = "False",
            )

        # Sites Subselection value
        sites_filter_value=arcpy.Parameter(
            displayName="Sites Filter Value",
            name="Sites_Filter_Value",
            datatype="String",
            parameterType="Optional",
            enabled = "False",
            )

        # Input Analysis Feature Classes
        analysis_features=arcpy.Parameter(
            displayName="Analysis Feature Classes",
            name="Analysis_Feature_Classes",
            datatype="GPValueTable",
            )
        analysis_features.columns = [['DEFeatureClass', 'Feature Class'], ['GPString', 'Field']]
        analysis_features.filters[1].type = 'ValueList'
        analysis_features.filters[1].list = ['Field']

        # Input Analysis Rasters
        analysis_rasters=arcpy.Parameter(
            displayName="Analysis Rasters",
            name="Analysis_Rasters",
            datatype="DERasterDataset",
            multiValue=True,
            )

        # Sample point spacing
        sample_spacing=arcpy.Parameter(
            displayName="Sample Point Spacing (m)",
            name="Sample_Point_Spacing",
            datatype="Long",
            )

        # N-trees
        n_trees=arcpy.Parameter(
            displayName="Number of Trees",
            name="Number_of_Trees",
            datatype="Long",
            )

        # Max depth
        max_depth=arcpy.Parameter(
            displayName="Maximum Tree Depth",
            name="Maximum_Tree_Depth",
            datatype="Long",
            )

        # Sample sizes
        sample_prop=arcpy.Parameter(
            displayName="Bootstrap Sample Proportion",
            name="Bootstrap_Sample_Proportion",
            datatype="Double",
            )

        # Min samples per split
        min_split=arcpy.Parameter(
            displayName="Minimum Samples per Split",
            name="Minimum_Samples_per_Split",
            datatype="Long",
            )

        # N-folds
        k_folds=arcpy.Parameter(
            displayName="Cross-Validation Folds",
            name="Cross_Validation_Folds",
            datatype="Long",
            )

        # Validation Test Proportion
        test_proportion=arcpy.Parameter(
            displayName="Final Validation Test Proportion",
            name="Test_Proportion",
            datatype="Double",
            )

        # Random seed
        random_seed=arcpy.Parameter(
            displayName="Random Seed",
            name="Random_Seed",
            datatype="Long",
            )

        # Output Location
        output_folder=arcpy.Parameter(
            displayName="Output Folder",
            name="Output_Folder",
            datatype="DEFolder",
            )

        return [
            input_analysis_area,
            surveys, surveys_filter, surveys_filter_field, surveys_filter_value,
            sites, sites_filter, sites_filter_field, sites_filter_value, analysis_features, analysis_rasters,
            sample_spacing, n_trees, max_depth, sample_prop, min_split, k_folds, test_proportion, random_seed,
            output_folder,
            ]


    def isLicensed(self):
        return True


    def updateParameters(self, params):
        (input_analysis_area,
        surveys, surveys_filter, surveys_filter_field, surveys_filter_value,
        sites, sites_filter, sites_filter_field, sites_filter_value, analysis_features, analysis_rasters,
        sample_spacing, n_trees, max_depth, sample_prop, min_split, k_folds, test_proportion, random_seed,
        output_folder,
        ) = params

        input_analysis_area.filter.list = ["Polygon"]
        surveys.filter.list = ["Polygon"]
        sites.filter.list = ["Polygon"]

        # Subselect Surveys
        if surveys.value:
            surveys_filter.enabled = "True"
        else:
            surveys_filter.enabled = "False"

        if surveys_filter.value:
            surveys_fields = arcpy.Describe(surveys.value).fields
            surveys_field_list = [f.name for f in surveys_fields if f.type in ["String", "Integer", "SmallInteger"]]
            surveys_filter_field.enabled = "True"
            surveys_filter_field.filter.type = "ValueList"
            surveys_filter_field.filter.list = surveys_field_list
        else:
            surveys_filter_field.value = ""
            surveys_filter_field.enabled = "False"

        if surveys_filter_field.value:
            surveys_vals = set([r[0] for r in arcpy.da.SearchCursor(surveys.value, surveys_filter_field.value)])
            surveys_filter_value.enabled = True
            surveys_filter_value.filter.type = "ValueList"
            surveys_filter_value.filter.list = sorted([v for v in surveys_vals if v])
        else:
            surveys_filter_value.value = ""
            surveys_filter_value.enabled = False

        # Subselect Sites
        if sites.value:
            sites_filter.enabled = "True"
        else:
            sites_filter.enabled = "False"

        if sites_filter.value:
            sites_fields = arcpy.Describe(sites.value).fields
            sites_field_list = [f.name for f in sites_fields if f.type in ["String", "Integer", "SmallInteger"]]
            sites_filter_field.enabled = "True"
            sites_filter_field.filter.type = "ValueList"
            sites_filter_field.filter.list = sites_field_list
        else:
            sites_filter_field.value = ""
            sites_filter_field.enabled = "False"

        if sites_filter_field.value:
            sites_vals = set([r[0] for r in arcpy.da.SearchCursor(sites.value, sites_filter_field.value)])
            sites_filter_value.enabled = True
            sites_filter_value.filter.type = "ValueList"
            sites_filter_value.filter.list = sorted([v for v in sites_vals if v])
        else:
            sites_filter_value.value = ""
            sites_filter_value.enabled = False

        # Filter analysis input fields
        if analysis_features.value and analysis_features.altered:
            analysis_fields = []
            for fc, _ in analysis_features.values:
                analysis_fields.extend(
                    [f.name for f in arcpy.Describe(fc).fields if f.type in ["String", "Integer", "SmallInteger"]]
                    )
            analysis_fields = list(set(analysis_fields))
            analysis_fields.sort()
            analysis_features.filters[1].list = analysis_fields

        if not sample_spacing.altered: sample_spacing.value = 100
        if not n_trees.altered: n_trees.value = 10
        if not max_depth.altered: max_depth.value = 10
        if not sample_prop.altered: sample_prop.value = 0.2
        if not min_split.altered: min_split.value = 10
        if not k_folds.altered: k_folds.value = 5
        if not test_proportion.altered: test_proportion.value = 0.2
        if not random_seed.altered: random_seed.value = 42


    def updateMessages(self, params):
        (input_analysis_area,
        surveys, surveys_filter, surveys_filter_field, surveys_filter_value,
        sites, sites_filter, sites_filter_field, sites_filter_value, analysis_features, analysis_rasters,
        sample_spacing, n_trees, max_depth, sample_prop, min_split, k_folds, test_proportion, random_seed,
        output_folder,
        ) = params

        if analysis_features.altered:
            for fc, val in analysis_features.values:
                if val not in [f.name for f in arcpy.Describe(fc).fields]:
                    error_msg = ("Field [{}] not found in feature class [{}]".format(val, os.path.basename(str(fc))))
                    analysis_features.setErrorMessage(error_msg)

            fc_paths = [str(path) for path, _ in analysis_features.values]
            fc_fields = [str(field) for _, field in analysis_features.values]

            if len(set(fc_paths)) != len(fc_paths):
                err_mg = ("Duplicate input feature classes: the same feature class cannnot be used more than once.")
                analysis_features.setErrorMessage(err_mg)
            if len(set(fc_fields)) != len(fc_fields):
                err_mg = ("Duplicate input feature classes field names: the same feature class field name cannnot be used more than once.")
                analysis_features.setErrorMessage(err_mg)

        if analysis_rasters.value:
            raster_paths = [str(path) for path in analysis_rasters.valueAsText.split(';')]
            if len(set(raster_paths)) != len(raster_paths):
                err_mg = ("Duplicate input rasters: the same raster cannnot be used more than once.")
                analysis_rasters.setErrorMessage(err_mg)

        if sample_prop.value and sample_prop.value >=1:
            err_mg = ("Bootstrap Sample Proportion must be a number between 0 and 1.")
            sample_prop.setErrorMessage(err_mg)

        if test_proportion.value and test_proportion.value >=0.5:
            err_mg = ("Validation Test Proportion must be a number between 0 and 0.5")
            sample_prop.setErrorMessage(err_mg)

        return


    def execute(self, params, messages):
        (input_analysis_area,
        surveys, surveys_filter, surveys_filter_field, surveys_filter_value,
        sites, sites_filter, sites_filter_field, sites_filter_value, analysis_features, analysis_rasters,
        sample_spacing, n_trees, max_depth, sample_prop, min_split, k_folds, test_proportion, random_seed,
        output_folder,
        ) = params

        for param in params:
            arcpy.AddMessage(f'{param.name}: {param.value}')

        # TODO: Make sure at least one feature or raster supplied

        try:
            #--- Create Geodatabase ----------------------------------------------------------------------------------------------

            # Date/time stamp for outputs
            dt_stamp = re.sub('[^0-9]', '', str(datetime.datetime.now())[:16])

            # Output fGDB name and full path
            file_name = os.path.splitext(os.path.basename(__file__))[0] # Name outputs with tool name and date
            gdb_name = "{}_{}".format(file_name, dt_stamp)
            gdb_path = os.path.join(output_folder.valueAsText, gdb_name + '.gdb')

            # Create a geodatabase
            arcpy.CreateFileGDB_management(output_folder.valueAsText, gdb_name, "10.0")

            # Set workspace to fGDB
            arcpy.env.workspace = gdb_name

            # Get input analysis area spatial reference for outputs
            spatial_ref = arcpy.Describe(input_analysis_area.value).spatialReference

            #--- Get Anaysis Area -------------------------------------------------------------------------------------

            # Copy analysis area and get acres
            analysis_area = os.path.join(gdb_path, 'Analysis_Area')
            arcpy.CopyFeatures_management(input_analysis_area.value, analysis_area)
            analysis_acres_total = get_acres(analysis_area)['Acres_Sum']
            print('\nTotal acres within analysis area: {}\n'.format(round(analysis_acres_total, 4)))

            #--- Get Survey Data --------------------------------------------------------------------------------------

            # Check for a subselection - must supply all three values!
            if (surveys_filter.value and surveys_filter_field.value and surveys_filter_value.value):
                where = build_where_clause(surveys.value, surveys_filter_field.value, [surveys_filter_value.value])
                tmp = arcpy.management.MakeFeatureLayer(surveys.value, "memory\\surveys", where)
            else:
                tmp = arcpy.management.MakeFeatureLayer(surveys.value, "memory\\surveys")

            # Clip to analysis area
            arcpy.analysis.Clip("memory\\surveys", analysis_area, "memory\\clipped_surveys")

            # Dissolve and get survey acreage
            analysis_surveys = os.path.join(gdb_path, 'Analysis_Surveys')
            arcpy.Dissolve_management("memory\\clipped_surveys", analysis_surveys)
            survey_acres_total = get_acres(analysis_surveys)['Acres_Sum']

            survey_coverage = survey_acres_total / analysis_acres_total
            arcpy.AddMessage('Survey acres within analysis area: {}'.format(round(survey_acres_total, 2)))
            arcpy.AddMessage('Survey proportion within analysis area: {}\n'.format(round(survey_coverage, 3)))

            # Enforce minimum survey coverage for analysis
            if survey_coverage < 0.05:
                raise InsufficientSurveyCoverage

            arcpy.management.Delete("memory\\surveys")
            arcpy.management.Delete("memory\\clipped_surveys")

            #--- Get Site Data ----------------------------------------------------------------------------------------

            # Check for a subselection - must supply all three values!
            if (sites_filter.value and sites_filter_field.value and sites_filter_value.value):
                where = build_where_clause(sites.value, sites_filter_field.value, [sites_filter_value.value])
                tmp = arcpy.management.MakeFeatureLayer(sites.value, "memory\\sites", where)
            else:
                tmp = arcpy.management.MakeFeatureLayer(sites.value, "memory\\sites")

            # Clip to surveyed coverage
            analysis_sites =  os.path.join(gdb_path, 'Analysis_Sites')
            arcpy.analysis.Clip("memory\\sites", analysis_surveys, analysis_sites)

            site_count = int(arcpy.GetCount_management(analysis_sites).getOutput(0))
            site_density = round(site_count/survey_acres_total, 4)
            acres_per_site = round(1/site_density, 2)

            arcpy.AddMessage('Sites identified for analysis: {}'.format(site_count))
            arcpy.AddMessage('Site density in surveyed areas (sites/acre): {}'.format(site_density))
            arcpy.AddMessage('Approximately 1 site every {} acres'.format(acres_per_site))

            if site_count < 30:
                raise InsufficientSiteSample

            arcpy.management.Delete("memory\\sites")

            #--- Create Sample Datasets -------------------------------------------------------------------------------

            # Get the analysis area extent coordinates
            desc = arcpy.Describe(analysis_area)
            xmin = desc.extent.XMin
            xmax = desc.extent.XMax
            ymin = desc.extent.YMin
            ymax = desc.extent.YMax

            # Create fishnet
            fishnet_poly = os.path.join(gdb_path, 'Fishnet')
            fishnet_points = os.path.join(gdb_path, 'Fishnet_label')
            sample_points = os.path.join(gdb_path, 'Sample_Points')
            arcpy.management.CreateFishnet(
                out_feature_class=fishnet_poly,
                origin_coord="{} {}".format(xmin, ymin),  # lower left
                y_axis_coord="{} {}".format(xmin, ymin+10),  # lower left + 10 meters to orient true north
                cell_width=sample_spacing.value,
                cell_height=sample_spacing.value,
                corner_coord="{} {}".format(xmax, ymax),  # upper right
                labels=True,
                template=analysis_area,
                geometry_type="POLYGON"
                )

            arcpy.analysis.Clip(fishnet_points, analysis_area, sample_points)
            arcpy.management.Delete(fishnet_poly)
            arcpy.management.Delete(fishnet_points)

            site_points = os.path.join(gdb_path, 'Site_Points')

            # Over sample the sites to ensure coverage - class balancing
            arcpy.management.CreateRandomPoints(
                out_path=gdb_path,
                out_name=os.path.basename(site_points),
                constraining_feature_class=analysis_sites,
                minimum_allowed_distance="{} Meters".format(np.sqrt(sample_spacing.value)),
                )

            # Append site oversample to sample points and delete
            arcpy.management.Append(site_points, sample_points, 'NO_TEST')
            arcpy.management.Delete(site_points)

        #--- Data Attribution -----------------------------------------------------------------------------------------

            # Feature Classes ---------------------------------------------
            if analysis_features.value:

                analysis_feature_classes, analysis_feature_fields = zip(*analysis_features.values)
                analysis_points = os.path.join(gdb_path, 'Analysis_Points')

                arcpy.Intersect_analysis(
                    in_features=analysis_feature_classes + (sample_points,),
                    out_feature_class=analysis_points,
                    )

                arcpy.management.Delete(sample_points)

                # clean up fields
                for field in arcpy.ListFields(analysis_points):
                    if field.name not in analysis_feature_fields:
                        try:
                            arcpy.DeleteField_management(analysis_points, field.name)
                        except:
                            pass

            # Rasters -----------------------------------------------------
            if analysis_rasters.value:
                arcpy.sa.ExtractMultiValuesToPoints(analysis_points, analysis_rasters.valueAsText.split(';'))

        #--- Isolate test/train and prediction datasets ---------------------------------------------------------------

            # Add class ID field
            class_field = 'Site_Point'
            arcpy.management.AddField(analysis_points, class_field, 'Short')

            # Extract the prediction points
            prediction_points = os.path.join(gdb_path, 'Prediction_Points')
            analysis_points_lyr = arcpy.management.MakeFeatureLayer(analysis_points, 'memory\\analysis_points')
            arcpy.management.SelectLayerByLocation(analysis_points_lyr, 'INTERSECT', analysis_surveys, invert_spatial_relationship=True)
            arcpy.management.CopyFeatures(analysis_points_lyr, prediction_points)
            arcpy.management.DeleteFeatures(analysis_points_lyr)

            # Encode site points
            arcpy.management.SelectLayerByLocation(analysis_points_lyr, 'INTERSECT', analysis_sites)
            arcpy.management.CalculateField(analysis_points_lyr, class_field, '1')
            arcpy.management.SelectLayerByLocation(analysis_points_lyr, 'INTERSECT', analysis_sites, invert_spatial_relationship=True)
            arcpy.management.CalculateField(analysis_points_lyr, class_field, '0')

            arcpy.management.Delete(analysis_points_lyr)

        #--- Random Forest --------------------------------------------------------------------------------------------

            # Prep the dataframe
            df = feature_class_to_data_frame(analysis_points)
            df.drop(columns=['Shape'], inplace=True)

            # Try to coerce strings to numbers
            string_columns = df.select_dtypes(include=['object', 'category']).columns
            df[string_columns] = df[string_columns].apply(pd.to_numeric, errors='ignore', axis=1)

            # One hot remaining string columns
            one_hot_columns = df.select_dtypes(include=['object', 'category']).columns
            df = one_hot_dataframe(df, one_hot_columns)

            # Create the Train / Test split
            test_df = df.sample(frac=test_proportion.value, random_state=random_seed.value)
            train_df = df.drop(test_df.index)

            train_df.to_csv(os.path.join(output_folder.valueAsText, 'Train_df.csv'))
            test_df.to_csv(os.path.join(output_folder.valueAsText, 'Test_df.csv'))

            train_dataset = train_df.values.tolist()

            test_actuals = test_df[class_field].values.tolist()
            test_df.drop(columns=[class_field], inplace=True)
            test_dataset = test_df.values.tolist()

            # TODO: Parameritize this??
            n_features = int(math.sqrt(len(train_dataset[0])-1))

            # TODO: make this optional
            # K-Fold cross-validation
            cross_validation_scores = random_forest.cross_validate(
                train_dataset,
                k_folds.value,
                max_depth.value,
                min_split.value,
                sample_prop.value,
                n_trees.value,
                n_features
                )
            arcpy.AddMessage(f'Cross-Validation Scores: f{cross_validation_scores}')

            # Build the final model on the whole training set
            final_model, final_predictions = random_forest(
                train_dataset,
                test_dataset,
                max_depth,
                min_size,
                sample_proportion,
                n_trees,
                n_features,
                )
            # Final accuracy metric
            confusion_matrix = random_forest.confusion_matrix(test_actuals, final_predictions)

            arcpy.AddMessage(confusion_matrix)

            # Predict the unsurveyed areas
            prediction_points_df = feature_class_to_data_frame(prediction_points)
            prediction_points_df.drop(columns=[class_field], inplace=True)
            prediction_dataset = prediction_points_df.values.tolist()
            predictions = predict_dataset(final_model, prediction_dataset)

            # join predictions to original point data
            with arcpy.da.UpdateCursor(prediction_points, class_field) as cur:
                for ix, row in enumerate(cur):
                    row = predictions[ix]
                    cur.updateRow(row)


        #--- Error Handling -------------------------------------------------------------------------------------------

        except InsufficientSurveyCoverage:
            msg = (
                "Insufficient survey coverage in analysis area.\n"
                "Model requires a minimum of 5 percent survey coverage for analysis.\n"
                "[exiting program]")
            arcpy.AddError(msg)

        except InsufficientSiteSample:
            msg = (
                "Too few sites in analysis area.\n"
                "Model requires a minimum of 30 sites for analysis.\n"
                "[exiting program]")
            arcpy.AddError(msg)

        except:
            # [Rails]  <-- -->  [Car]
            err = str(traceback.format_exc())
            arcpy.AddError(err)

        finally:
            arcpy.management.Delete('memory')

        return
