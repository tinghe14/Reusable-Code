def get_pmid_records_from_file(pmid_filename):
    pmid_records = []
    with open(pmid_filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader) # Skip header row
        for row in csvreader:
            pmid_records.append({'pmid_number': row[0], 'title': row[1], 'abstract': row[2]})
    return pmid_records
