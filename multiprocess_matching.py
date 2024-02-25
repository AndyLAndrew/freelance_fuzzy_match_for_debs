import pandas as pd
import multiprocessing  # for high speed action!
from polyfuzz import PolyFuzz




# need to convert NaNs to 0 so the matcher can accept all character types
def fillna_col(column_list, final_clean):
    for name in column_list:
        final_clean[name].fillna('0', inplace=True)


def dictionary_storage_state(final_clean, agency_ref):
    unmatched_dict = {}
    ref_dict = {}

    for state in final_clean["State"].unique().tolist():  # for each state
        # for unmatched
        unmatched_dict[state] = []
        unmatched_dict[state].append(final_clean[final_clean["State"] == state])

        # for ref
        ref_dict[state] = []
        ref_dict[state].append(agency_ref[agency_ref["State"] == state])
    return ref_dict, unmatched_dict


# to conduct fuzzy matching, we will need to loop and rewrite using lists
# new dictionary needed
def fuzzy_match_process(agency_col, ref_dict, unmatched_dict, modeltype, threshold, final_clean):
    matched_dict = {}
    for state in final_clean["State"].unique().tolist():
        from_list = unmatched_dict[state][0][agency_col].tolist()  # generating lists
        to_list = ref_dict[state][0]["Agencies_involved"].tolist()

        # needed to add a null check as some states have 0 names to try unfortunately
        if not all(item == "0" for item in from_list):
            model = PolyFuzz(modeltype)  # initializing model
            model.match(from_list, to_list)  # begin matching
            match_dict = model.get_matches()  # grab matches

            # filter based on threshold
            templist = [b if similarity > threshold else a for a, b, similarity in
                        zip(match_dict["From"], match_dict["To"], match_dict["Similarity"])]
            matched_dict[state] = [unmatched_dict[state][0].index.tolist(), unmatched_dict[state][0]["State"].tolist(),
                                   templist, match_dict["Similarity"].tolist()]  # add back into dictionary for state
        else:
            matched_dict[state] = [unmatched_dict[state][0].index.tolist(), unmatched_dict[state][0]["State"].tolist(),
                                   from_list, from_list]
            # add back

    return matched_dict


def dataframe_generator(matched_dict, column_append_name, final_clean):
    matched_dataframe = pd.concat(
        [pd.DataFrame(matched_dict[key]).T.set_axis(['Col1', 'state', 'fuzzy_matched', 'similarity'], axis=1) for key in
         matched_dict], keys=matched_dict.keys())
    # Set 'Col1' as the index
    matched_dataframe.set_index('Col1', inplace=True)
    matched_dataframe.index.name = None  # Remove the index name
    matched_dataframe.sort_index(inplace=True)  # sort to keep the right order
    final_clean.sort_index(inplace=True)  # do the same to final_clean

    # adding cols to final_clean
    fuzzy_match_name = "fuzzy_matched" + "_" + column_append_name
    similarity_name = "similarity" + "_" + column_append_name
    final_clean[fuzzy_match_name] = matched_dataframe['fuzzy_matched']
    final_clean[similarity_name] = matched_dataframe['similarity']

    return final_clean


# lastly we need to do a light merge over of any missing Ori's
# this will take some time...
def merging_func(ori_col, fuzzy_matched_agency_col, similarity_col, final_clean, agency_ref, threshold):
    match, nomatch = 0, 0

    tracking = len(
        final_clean[final_clean[ori_col].isna()][[ori_col, 'State', fuzzy_matched_agency_col, similarity_col]])
    count_track = 0

    for index, row in final_clean[final_clean[ori_col].isna()][
        [ori_col, 'State', fuzzy_matched_agency_col, similarity_col]].iterrows():
        if count_track % 1000 == 0: print(f"working...{count_track} of {tracking} for {ori_col} at {threshold}")
        # print(row['Ori'], row['State'], row['fuzzy_matched'], row['similarity'])
        # we will now recursively search through all possible matches
        # there are better ways to do this, but for now, this will do to help reduce some NaNs
        # Check for exact match in agency_ref
        try:
            mask = (agency_ref['State'] == row['State']) & (
                        agency_ref['Agencies_involved'].str.upper() == row[fuzzy_matched_agency_col].upper())
            matching_rows = agency_ref[mask]
            if not matching_rows.empty:
                updated_ori_value = matching_rows.iloc[0]['Ori']
                match += 1
                final_clean.at[index, ori_col] = updated_ori_value
            else:
                nomatch += 1
            count_track += 1  # to keep track of progress
        except AttributeError:
            # Handle the case where 'State' or 'Agencies_involved' might be NaN
            # You can choose to print a message or take other actions as needed
            print(f"Error: NaN values found in 'State' or 'Agencies_involved' at index {index}")
    # Reporting progress
    print("Ori Matches Found: ", match, "\n Ori unmatched remaining: ", nomatch)
    print("Remaining missing observations:", final_clean[ori_col].isna().sum())
    return final_clean


# Putting it all together
def full_run(param_list):
    algorithm, thresh = param_list
    final_clean_fr = pd.read_csv("data/final_clean_subset.csv")
    agency_ref_fr = pd.read_csv("data/final_clean_dictionary.csv")

    # renaming columns in final_clean_dict
    agency_ref_fr.rename(columns={'State': 'State', 'Ori_split8': 'Ori', 'Agency8': 'Agencies_involved'},
                            inplace=True)

    # generating the list of columns we wish to match to our agency reference
    column_list_for_func = ["Agency" + str(i) for i in range(1, 9)]

    # switching nans to 0
    fillna_col(column_list_for_func, final_clean_fr)

    # dictionary to store each state for individual fuzzy matching
    ref, nonmatch = dictionary_storage_state(final_clean_fr, agency_ref_fr)

    # conducting fuzzy matching
    list_of_agency_dicts = [
        fuzzy_match_process(agency_col=x, ref_dict=ref, unmatched_dict=nonmatch, modeltype=algorithm, threshold=thresh,
                            final_clean=final_clean_fr) for x in column_list_for_func]

    # generate new dataframes
    for agency_dict, column_name in zip(list_of_agency_dicts, column_list_for_func):
        final_clean_fr = dataframe_generator(agency_dict, column_name, final_clean_fr)

    # generating lists of column names to iterate through
    fuzzy_match_list = ["fuzzy_matched_Agency" + str(i) for i in range(1, 9)]
    ori_col_list = ["Ori_split" + str(i) for i in range(1, 9)]
    similarity_col_list = ["similarity_Agency" + str(i) for i in range(1, 9)]

    # running function
    for ori, fuzzy, simm in zip(ori_col_list, fuzzy_match_list, similarity_col_list):
        final_clean_fr = merging_func(ori, fuzzy, simm, final_clean_fr, agency_ref_fr, threshold=thresh)

    # saving to csv
    string_temp = "matched_replaced_" + str(int(thresh * 100)) + "_tfidf.csv"
    final_clean_fr.to_csv(string_temp)

    # cleaning up
    del final_clean_fr, agency_ref_fr

def maintain_workers(queue, jobs):
    while True:
        core = queue.get()
        if core is None:
            break
        jobs[core].start()
        jobs[core].join()
        queue.put(core)

def main():
    # pulling number of cores to send jobs to
    num_cores = (
        multiprocessing.cpu_count()
    )
    available_cores = multiprocessing.Queue()
    # readying cores
    for _ in range(num_cores):
        available_cores.put(None)
    print(f"Machine has {num_cores}")

    # generating our lists of params
    grid_search_list = [["TF-IDF", 0.7],
                        ["TF-IDF", 0.8],
                        ["TF-IDF", 0.9],
                        ["TF-IDF", 0.95],
                        ["TF-IDF", 1.0]]

    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(full_run, grid_search_list)


if __name__ == "__main__":
    main()
    exit()
