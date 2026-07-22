# Phase 17 normalization rules — evidence table

Corpus: 7,413 `.cleaned.txt`. Mixed-script token types: 587, 3234 occurrences.

Every target below is attested in the corpus. Nothing is guessed.

| class | corrupt form | → target | src occ | src files | target occ | status |
|---|---|---|---|---|---|---|
| loanword | `मEDITATION` | `मेडिटेशन` | 1498 | 417 | 9377 | APPLIED |
| native | `रishi` | `रिशी` | 579 | 251 | 302 | APPLIED |
| scatter | `मीडिटेशन` | `मेडिटेशन` | 501 | 0 | 9377 | APPLIED |
| scatter | `मैडिटेशन` | `मेडिटेशन` | 404 | 0 | 9377 | APPLIED |
| scatter | `मेंडिटेशन` | `मेडिटेशन` | 316 | 0 | 9377 | APPLIED |
| loanword | `सittings` | `सिटिंग्स` | 80 | 34 | 11 | APPLIED |
| scatter | `मेडिटीशन` | `मेडिटेशन` | 54 | 0 | 9377 | APPLIED |
| native | `निभaya` | `निभाया` | 35 | 22 | 98 | APPLIED |
| loanword | `मEDITेशन` | `मेडिटेशन` | 33 | 16 | 9377 | APPLIED |
| scatter | `मिडिटेशन` | `मेडिटेशन` | 24 | 0 | 9377 | APPLIED |
| native | `रindo` | `रिंदो` | 23 | 17 | 22 | APPLIED |
| native | `बaki` | `बाकी` | 20 | 15 | 1343 | APPLIED |
| loanword | `थॉUGHT` | `थॉट` | 19 | 14 | 279 | APPLIED |
| native | `Kumarों` | `कुमारों` | 18 | 11 | 194 | APPLIED |
| scatter | `मेडिटिशन` | `मेडिटेशन` | 18 | 0 | 9377 | APPLIED |
| loanword | `थॉUGHTS` | `थॉट्स` | 17 | 10 | 387 | APPLIED |
| native | `तirth` | `तीर्थ` | 15 | 10 | 472 | APPLIED |
| native | `कismet` | `किस्मत` | 14 | 6 | 216 | APPLIED |
| native | `इनayat` | `इनायत` | 14 | 9 | 95 | APPLIED |
| loanword | `मeditation` | `मेडिटेशन` | 13 | 13 | 9377 | APPLIED |
| native | `बazaar` | `बाज़ार` | 13 | 6 | 24 | APPLIED |
| native | `घबरaya` | `घबराया` | 10 | 1 | 32 | APPLIED |
| loanword | `मEDITATORS` | `मेडिटेटर्स` | 9 | 8 | 138 | APPLIED |
| native | `रishियों` | `रिशियों` | 9 | 4 | 91 | APPLIED |
| loanword | `मENTAL` | `मेंटल` | 6 | 4 | 547 | APPLIED |
| native | `रishiओं` | `रिशियों` | 6 | 3 | 91 | APPLIED |
| loanword | `मैनifest` | `मैनिफेस्ट` | 5 | 5 | 19 | APPLIED |
| native | `रishiयों` | `रिशियों` | 4 | 3 | 91 | APPLIED |
| native | `सद्गuru` | `सद्गुरु` | 4 | 2 | 1367 | APPLIED |
| loanword | `वIBRATION` | `वाइब्रेशन` | 3 | 3 | 1341 | APPLIED |
| loanword | `ब्रेathing` | `ब्रीदिंग` | 3 | 3 | 54 | APPLIED |
| native | `रindों` | `रिंदों` | 2 | 2 | 229 | APPLIED |
| native | `कलiforniya` | `कैलिफोर्निया` | 2 | 2 | 22 | APPLIED |
| native | `रind` | `रिंद` | 1 | 1 | 438 | APPLIED |
| native | `गुरुनanak` | `गुरुनानक` | 1 | 1 | 120 | APPLIED |
| native | `प्रahalad` | `प्रहलाद` | 1 | 1 | 440 | APPLIED |
| native | `कumbhakaran` | `कुंभकरण` | 1 | 1 | 3 | target unattested |
| native | `कलifornia` | `कैलिफोर्निया` | 1 | 1 | 22 | APPLIED |
| native | `बदरिनath` | `बद्रीनाथ` | 1 | 1 | 7 | APPLIED |

## Flagged — NOT rewritten (human decision required)

| form | occ | why |
|---|---|---|
| `महारishi` | 1 | target ambiguous: महर्षी (127) is the standard title, महारिशी is literal |
| `रishiRai` | 1 | personal name; correct Devanagari unknown |
| `रishiyanand` | 2 | personal name; correct Devanagari unknown |
| `प्रमरishi` | 1 | unclear compound |
| `कishkinda` | 4 | किष्किंधा has 0 corpus occurrences; cannot verify intended spelling |
| `दधichi` | 2 | दधीचि has 0 corpus occurrences; cannot verify intended spelling |
| `अरindo` | 1 | = 'और रिंद' elided; two-token target, not a name |
| `रामतirth` | 2 | राम तीर्थ is two tokens; whole-token map cannot emit a space |
| `रामतirthों` | 1 | राम तीर्थों is two tokens |
| `पrahलाद` | 1 | target प्रहलाद attested, but this form is 1 occ; low value |
| `रish्यों` | 1 | 1 occ; ambiguous target |
| `रishiजन` | 1 | two-token target रिशी जन |
| `रishiजान` | 1 | two-token target |
| `रishिकुमारों` | 1 | two-token target रिशी कुमारों |
| `तakte` | 10 | target ताकते unattested |
| `रindre` | 2 | target रिंद्रे unattested |

## Residual: 539 mixed-script types, 743 occurrences left unmapped (23.0%)

Top 25 unmapped:

| form | occ |
|---|---|
| `जागरण—enlightened` | 15 |
| `नहीं—available` | 15 |
| `है—mind` | 10 |
| `जeho` | 9 |
| `मeditate` | 9 |
| `रहना—mind` | 7 |
| `देkho` | 6 |
| `उनhone` | 6 |
| `हैं—we` | 6 |
| `गुंgeh` | 5 |
| `मEDIT` | 5 |
| `पिटiful` | 5 |
| `थॉughts` | 5 |
| `है—we` | 5 |
| `को—mind` | 5 |
| `संभavana` | 4 |
| `थoughts` | 4 |
| `हुशyar` | 4 |
| `सोभagy` | 4 |
| `कियacho` | 4 |
| `में—You` | 4 |
| `साanu` | 4 |
| `jagaए` | 3 |
| `उठaye` | 3 |
| `थकawat` | 3 |
