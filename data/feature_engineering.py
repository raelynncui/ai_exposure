"""
Pipeline step 3: feature engineering

Reads data/joined_cleaned.csv and produces data/features.csv with ~30 clean interpretable features
"""

from pathlib import Path
import polars as pl

INPUT  = Path("data/joined_cleaned.csv")
OUTPUT = Path("data/features.csv")

df = pl.read_csv(INPUT, infer_schema_length=2000, null_values=[""])

def share(num: pl.Expr, denom: pl.Expr) -> pl.Expr:
    return pl.when(denom > 0).then(num / denom).otherwise(None)

# Category 1: Occupation shares (C24010) 
# C24010 only has male/female counts — sum them for each of the 5 groups.
# Denominator = "Total" (total civilian employed population 16+).
# Drop pct_production_transport as reference category (5 groups sum to 1).

occ_denom = pl.col("Total")

occ_features = [
    share(
        pl.col("Total: Male: Management, Business, Science, And Arts Occupations")
        + pl.col("Total: Female: Management, Business, Science, And Arts Occupations"),
        occ_denom
    ).alias("pct_mgmt_business_arts"),

    share(
        pl.col("Total: Male: Service Occupations")
        + pl.col("Total: Female: Service Occupations"),
        occ_denom
    ).alias("pct_service"),

    share(
        pl.col("Total: Male: Sales And Office Occupations")
        + pl.col("Total: Female: Sales And Office Occupations"),
        occ_denom
    ).alias("pct_sales_office"),

    share(
        pl.col("Total: Male: Natural Resources, Construction, And Maintenance Occupations")
        + pl.col("Total: Female: Natural Resources, Construction, And Maintenance Occupations"),
        occ_denom
    ).alias("pct_natural_resources_construction"),
    # pct_production_transport omitted as reference category
]

# Category 2: Industry shares (C24030)
# C24030 also only has male/female — same approach.
# Denominator = "Total (ORG_C24030001)".
# Drop Other Services as reference category.

ind_denom = pl.col("Total (ORG_C24030001)")

def ind(m: str, f: str, alias: str) -> pl.Expr:
    return share(pl.col(m) + pl.col(f), ind_denom).alias(alias)

ind_features = [
    ind("Total: Male: Agriculture, Forestry, Fishing And Hunting, And Mining",
        "Total: Female: Agriculture, Forestry, Fishing And Hunting, And Mining",
        "pct_industry_agriculture"),
    ind("Total: Male: Construction",
        "Total: Female: Construction",
        "pct_industry_construction"),
    ind("Total: Male: Manufacturing",
        "Total: Female: Manufacturing",
        "pct_industry_manufacturing"),
    ind("Total: Male: Wholesale Trade",
        "Total: Female: Wholesale Trade",
        "pct_industry_wholesale"),
    ind("Total: Male: Retail Trade",
        "Total: Female: Retail Trade",
        "pct_industry_retail"),
    ind("Total: Male: Transportation And Warehousing, And Utilities",
        "Total: Female: Transportation And Warehousing, And Utilities",
        "pct_industry_transport_utilities"),
    ind("Total: Male: Information",
        "Total: Female: Information",
        "pct_industry_information"),
    ind("Total: Male: Finance And Insurance, And Real Estate, And Rental And Leasing",
        "Total: Female: Finance And Insurance, And Real Estate, And Rental And Leasing",
        "pct_industry_finance"),
    ind("Total: Male: Professional, Scientific, And Management, And Administrative, And Waste Management Services",
        "Total: Female: Professional, Scientific, And Management, And Administrative, And Waste Management Services",
        "pct_industry_professional"),
    ind("Total: Male: Educational Services, And Health Care And Social Assistance",
        "Total: Female: Educational Services, And Health Care And Social Assistance",
        "pct_industry_education_healthcare"),
    ind("Total: Male: Arts, Entertainment, And Recreation, And Accommodation And Food Services",
        "Total: Female: Arts, Entertainment, And Recreation, And Accommodation And Food Services",
        "pct_industry_arts_food"),
    ind("Total: Male: Public Administration",
        "Total: Female: Public Administration",
        "pct_industry_public_admin"),
    # pct_industry_other_services omitted as reference category
]

# Category 3: Class of worker shares (B24080)
# B24080 also only has male/female — sum them.
# Denominator = "Total (ORG_B24080001)".
# Government = local + state + federal combined.
# Drop unpaid family workers (near-zero).

cow_denom = pl.col("Total (ORG_B24080001)")

cow_features = [
    share(
        pl.col("Total: Male: Private For-Profit Wage And Salary Workers")
        + pl.col("Total: Female: Private For-Profit Wage And Salary Workers"),
        cow_denom
    ).alias("pct_private_forprofit"),

    share(
        pl.col("Total: Male: Private For-Profit Wage And Salary Workers: Self-Employed In Own Incorporated Business Workers")
        + pl.col("Total: Female: Private For-Profit Wage And Salary Workers: Self-Employed In Own Incorporated Business Workers"),
        cow_denom
    ).alias("pct_self_employed_incorporated"),

    share(
        pl.col("Total: Male: Private Not-For-Profit Wage And Salary Workers")
        + pl.col("Total: Female: Private Not-For-Profit Wage And Salary Workers"),
        cow_denom
    ).alias("pct_nonprofit"),

    share(
        pl.col("Total: Male: Local Government Workers")    + pl.col("Total: Female: Local Government Workers")
        + pl.col("Total: Male: State Government Workers")  + pl.col("Total: Female: State Government Workers")
        + pl.col("Total: Male: Federal Government Workers") + pl.col("Total: Female: Federal Government Workers"),
        cow_denom
    ).alias("pct_government"),

    share(
        pl.col("Total: Male: Self-Employed In Own Not Incorporated Business Workers")
        + pl.col("Total: Female: Self-Employed In Own Not Incorporated Business Workers"),
        cow_denom
    ).alias("pct_self_employed_unincorporated"),
    # pct_unpaid_family omitted (near-zero, reference category)
]

# Category 4: Education shares (B15003) 
# B15003 has combined totals. Denominator = population 25+.
# Collapse 24 grade levels into 3 tiers (drop less-than-HS as reference).

edu_denom = pl.col("Total (ORG_B15003001)")

edu_features = [
    share(
        pl.col("Total: Regular High School Diploma") + pl.col("Total: Ged Or Alternative Credential"),
        edu_denom
    ).alias("pct_hs_or_ged"),

    share(
        pl.col("Total: Some College Less Than 1 Year")
        + pl.col("Total: Some College 1 Or More Years No Degree")
        + pl.col("Total: Associate'S Degree"),
        edu_denom
    ).alias("pct_some_college_assoc"),

    share(
        pl.col("Total: Bachelor'S Degree")
        + pl.col("Total: Master'S Degree")
        + pl.col("Total: Professional School Degree")
        + pl.col("Total: Doctorate Degree"),
        edu_denom
    ).alias("pct_bachelors_plus"),
    # pct_less_than_hs omitted as reference category
]

# Category 5: Income 
income_features = [
    pl.col("Median Household Income In The Past 12 Months (In 2024 Inflation-Adjusted Dollars)")
      .alias("median_household_income"),
    pl.col("Per Capita Income In The Past 12 Months (In 2024 Inflation-Adjusted Dollars)")
      .alias("per_capita_income"),
]

# Category 6: Commute mode shares (B08301) 
commute_denom = pl.col("Total (ORG_B08301001)")

commute_features = [
    share(pl.col("Total: Car Truck Or Van: Drove Alone"), commute_denom).alias("pct_drove_alone"),
    share(pl.col("Total: Public Transportation"),         commute_denom).alias("pct_public_transit"),
    share(pl.col("Total: Worked From Home"),              commute_denom).alias("pct_worked_from_home"),
]

# Category 7: Technology access 
tech_features = [
    share(pl.col("Total: Broadband Of Any Type"),
          pl.col("Total (ORG_B28002001)")).alias("pct_broadband"),

    share(pl.col("Total: Has One Or More Types Of Computing Devices"),
          pl.col("Total (ORG_B28001001)")).alias("pct_has_computer"),
]

# Category 8: Long commute share (B08303) 
travel_features = [
    share(
        pl.col("Total: 45 To 59 Minutes")
        + pl.col("Total: 60 To 89 Minutes")
        + pl.col("Total: 90 Or More Minutes"),
        pl.col("Total (ORG_B08303001)")
    ).alias("pct_long_commute"),
]

all_feature_exprs = (
    occ_features + ind_features + cow_features +
    edu_features + income_features + commute_features +
    tech_features + travel_features
)

df = df.with_columns(all_feature_exprs)

feature_cols = [e.meta.output_name() for e in all_feature_exprs]
features = df.select(["FIPS", "AI Exposure Score (0-0.29)"] + feature_cols)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
features = features.drop_nulls()
features.write_csv(OUTPUT)