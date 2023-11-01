import requests
import pandas as pd
import json

year_to_dataset = {
    2014: 'jt7v-77mi',
    2015: 'c284-tqph',
    2016 : 'kiv2-tbus',
    2017: '2bnn-yakx',
    2018: 'a5td-mswe',
    2019: 'faiq-9dfq',
    2020: 'p7t3-5i9s',
    2021: 'kvfd-bves',
    2022: '7mxj-7a6y',
    2023: 'pvqr-7yc4'
}

# year = int(input("Enter Year"))

url = f"https://data.cityofnewyork.us/resource/nc67-uf89.json?$query=SELECT%0A%20%20%60plate%60%2C%0A%20%20%60state%60%2C%0A%20%20%60license_type%60%2C%0A%20%20%60summons_number%60%2C%0A%20%20%60issue_date%60%2C%0A%20%20%60violation_time%60%2C%0A%20%20%60violation%60%2C%0A%20%20%60judgment_entry_date%60%2C%0A%20%20%60fine_amount%60%2C%0A%20%20%60penalty_amount%60%2C%0A%20%20%60interest_amount%60%2C%0A%20%20%60reduction_amount%60%2C%0A%20%20%60payment_amount%60%2C%0A%20%20%60amount_due%60%2C%0A%20%20%60precinct%60%2C%0A%20%20%60county%60%2C%0A%20%20%60issuing_agency%60%2C%0A%20%20%60violation_status%60%2C%0A%20%20%60summons_image%60%0AWHERE%0A%20%20caseless_contains(%60judgment_entry_date%60%2C%20%222023%22)%0A%20%20%20%20OR%20(caseless_contains(%60judgment_entry_date%60%2C%20%222022%22)%0A%20%20%20%20%20%20%20%20%20%20OR%20caseless_contains(%60judgment_entry_date%60%2C%20%222021%22))%0AORDER%20BY%20%60judgment_entry_date%60%20ASC%20NULL%20LAST%0ALIMIT%20103190%0AOFFSET%200&"

response = requests.get(url)


combined_data = pd.DataFrame({})

if response.status_code == 200:
    data = response.json()
    judgement_list = []

    for item in data:
        summons_number = item["summons_number"]
        year = item["judgment_entry_date"].split("/", 3)[2]
        dataset_id = year_to_dataset[int(year)]
        url = f"https://data.cityofnewyork.us/resource/{dataset_id}.json?$query=SELECT `issue_date`, `violation_code`, `vehicle_body_type`, `vehicle_make`, `issuing_agency`, `violation_location`, `violation_precinct`, `issuer_precinct`, `issuer_code`, `issuer_command`, `issuer_squad`, `violation_time`, `violation_county`,  `street_name`, `intersecting_street`, `law_section`, `sub_division`, `violation_legal_code`,`unregistered_vehicle`, `vehicle_year`, `meter_number`, `violation_post_code`, `violation_description`, `no_standing_or_stopping_violation`, `hydrant_violation`, `double_parking_violation` WHERE `summons_number` = {summons_number} ORDER BY `:id` ASC NULL LAST LIMIT 100 OFFSET 0"
        
#         url = f"https://data.cityofnewyork.us/api/id/{dataset_id}.json?$select=plate,state,license_type,summons_number,issue_date,violation_time,violation,judgment_entry_date,fine_amount,penalty_amount,interest_amount,reduction_amount,payment_amount,amount_due,precinct,county,issuing_agency,violation_status&$where=issue_date between '01/01/{year}' and '01/01/{year+1}' and judgment_entry_date is not null and reduction_amount < fine_amount&$order=judgment_entry_date+DESC&$offset=0"
        
        response = requests.get(url)

        if response.status_code == 200:
            summons_data = response.json()
            for summons_item in summons_data:
                for key, value in summons_item.items():
                    item[key] = value
            judgement_list.append(item)
        else:
            print(f"Failed to retrieve data for year {year} and summons {summons_number}. Status code:", response.status_code)

    combined_data = pd.DataFrame(judgement_list)

    # Save the combined data to a CSV file
    combined_data.to_csv(f"combined_data_.csv", index=False)
else:
    print("Failed to retrieve data. Status code:", response.status_code)

print("Data saved to CSV file.")