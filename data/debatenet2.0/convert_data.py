import jsonlines
import uuid 
from nltk.tokenize import sent_tokenize
from collections import Counter
import csv

# read annotated claims
claims = []
with jsonlines.open("./DebateNet-migr15v2.jsonl") as fp:
	for obj in fp:
		claims.append(obj)

# read TAZ articles
texts = []
with jsonlines.open("./DebateNet-migr15v2_taz.json") as fp:
	for obj in fp:
		texts.append(obj)
texts = texts[0]

dataset = []
# iterate over articles
# for each article, check if there exists annotations
# if yes: add annotation to the dataset as positive sample; remove annotation from article; split remaining sentences and add them to dataset as negative examples
for article_id, article_body in texts.items():
	related_claims = [claim for claim in claims if str(claim['doc_id']) == article_id]
	filtered_related_claims = []
	# several claims are contained multiple times in the dataset (with different political parties)
	for rel_claim in related_claims:
		if rel_claim['quote'] not in [i['quote'] for i in filtered_related_claims]:
			filtered_related_claims.append(rel_claim)

	indices = []
	date = None
	for cl in filtered_related_claims:
		dataset.append({
			'id': uuid.uuid4().hex,
			'text': cl['quote'],
			'tag': 'claim',
			'date': cl['cdate']
		})
		indices += list(range(cl['begin'], cl['end'] + 1))
		date = cl['cdate'] # saving date for article datapoints
	# remove annotated claims from article body
	article_body = "".join([char for idx, char in enumerate(article_body) if idx not in indices])
	# replace newlines with space (articles contain lots of newlines)
	article_body = article_body.replace("\n", " ")
	# create sentences from article body and add them to dataset as negative samples
	for sent in sent_tokenize(article_body):
		# quality check: should be longer than two tokens
		if len(sent) > 2:
			dataset.append({
				'id': uuid.uuid4().hex,
				'text': sent,
				'tag': 'noclaim',
				'date': date # all claims should have the same data and the date should be the one of the original article
			})

# write to output CSV file
with open('debatenet-migr15v2-binary-text-classification.csv', 'w') as fp:
	csvwriter = csv.writer(fp, delimiter=',', quotechar='"')
	csvwriter.writerow(['id', 'text', 'tag', 'date'])
	for d in dataset:
		csvwriter.writerow([d['id'], d['text'], d['tag'], d['date']])