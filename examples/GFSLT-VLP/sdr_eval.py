import pickle
import json
import numpy as np
from utils import tsne_visualize_bins 

binned_labels = None
# bin from GF_VLP
binned_labels = {'bin_1': ['WO', 'ICH', 'SELTEN', 'SECHSHUNDERT', 'DU', 'DAZWISCHEN', 'ALLE', 'VERAENDERN', 'WINTER', 'UNTER', 'HABEN', 'neg-HABEN', 'KURZ', 'GEFRIEREN', 'negalp-AUCH', 'MAI', 'ZEIGEN-BILDSCHIRM', 'VORSICHT', 'SEE', 'METER', 'ZONE', 'DURCHGEHEND', 'TAUEN', 'UNTERSCHIED', 'VORAUSSAGE', 'FEBRUAR'], 'bin_2': ['NAH', 'TATSAECHLICH', 'DARUM', 'WIE-IMMER', 'NOVEMBER', 'ALLGAEU', 'MACHEN', 'ERSTE', 'DAUER', 'DRITTE', 'ACHTE', 'WIEDER', 'TAGSUEBER', 'WAS', 'REST', 'WECHSEL', 'SCHEINEN', 'GLEICH', 'ZWEITE', 'GERADE', 'HABEN2', 'NULL', 'M', 'OFT', 'VORAUS', 'WENIGER'], 'bin_3': ['WENIG', 'negalp-KEIN', 'ABWECHSELN', 'JUNI', 'MITTWOCH', 'MITTEILEN', 'NACHMITTAG', 'LANGSAM', 'GRAUPEL', 'DREI', 'KOELN', 'BURG', 'INFORMIEREN', 'SKANDINAVIEN', 'EIN-PAAR-TAGE', 'FUENFTE', 'WOCHE', 'SINKEN', 'HIMMEL', 'SCHOTTLAND', 'BERLIN', 'FEUCHT', 'MAXIMAL', 'VOGEL', 'neg-REGEN', 'AUGUST'], 'bin_4': ['ZUERST', 'GRUND', 'WIE-AUSSEHEN', 'MITTAG', 'WEITER', 'AUFLOESEN', 'PLUS', 'LOCKER', 'SCHWER', 'BODEN', 'DIENSTAG', 'TSCHUESS', 'ZWEI', 'MAERZ', 'APRIL', 'OKTOBER', 'STERN', 'HOEHE', 'BESSER', 'DREISSIG', 'UNWETTER', 'WALD', 'JULI', 'LEICHT', 'VIERZEHN', 'HAGEL'], 'bin_5': ['VOR', 'FUENF', 'SIEBEN', 'DABEI', 'IN-KOMMEND', 'AEHNLICH', 'ORT', 'SPAETER', 'BAYERN', 'HERBST', 'KUEHL', 'VIER', 'STRASSE', 'FROST', 'ZWISCHEN', 'MAL', 'ACHT', 'MONTAG', 'BRAND', 'EINS', 'DOCH', 'WIE', 'DREIZEHN', 'SACHSEN', 'SAMSTAG', 'DRUCK'], 'bin_6': ['NAECHSTE', 'GLATT', 'RUHIG', 'SCHNEIEN', 'NEUNZEHN', 'SEPTEMBER', 'SECHSZEHN', 'ZWOELF', 'SIEBZEHN', 'MIT', 'KALT', 'SPEZIELL', 'WAHRSCHEINLICH', 'SUEDOST', 'UEBERWIEGEND', 'SUEDWEST', 'BLEIBEN', 'NEUN', 'DESHALB', 'TEIL', 'BEGRUESSEN', 'NORDOST', 'STARK', 'DEZEMBER', 'NACH', 'WENN'], 'bin_7': ['MEER', 'BIS', 'WUENSCHEN', 'DEUTSCHLAND', 'MINUS', 'MISCHUNG', 'VERSCHIEDEN', 'WECHSELHAFT', 'ZUSCHAUER', 'HAUPTSAECHLICH', 'GEFAHR', 'FUENFZEHN', 'SEHEN', 'DONNERSTAG', 'TRUEB', 'LUFT', 'ODER', 'ABER', 'FRUEH', 'DAZU', 'SO', 'UEBER', 'TEMPERATUR', 'FREITAG', 'ANFANG', 'SONNTAG'], 'bin_8': ['NEBEL', 'HEISS', 'STURM', 'GEWITTER', 'ORKAN', 'FLUSS', 'NUR', 'MAESSIG', 'MEISTENS', 'ELF', 'LIEB', 'UND', 'LANG', 'LAND', 'BEWOELKT', 'SCHAUER', 'FRISCH', 'MILD', 'ZWANZIG', 'BISSCHEN', 'SCHWACH', 'KUESTE', 'KOENNEN', 'WOCHENENDE', 'ALPEN', 'MOEGLICH'], 'bin_9': ['IM-VERLAUF', 'BERG', 'SECHS', 'SCHON', 'ENORM', 'ACHTZEHN', 'TROCKEN', 'ZEHN', 'GRAD', 'VERSCHWINDEN', 'WOLKE', 'SUED', 'JANUAR', 'EUROPA', 'WARM', 'JETZT', 'NEU', 'OST', 'TEILWEISE', 'poss-EUCH', 'WEST', 'AUCH', 'TIEF', 'WARNUNG', 'NORDWEST', 'REGEN'], 'bin_10': ['MORGEN', 'DEUTSCH', 'FREUNDLICH', 'WIND', 'KOMMEN', 'NORD', 'VIEL', 'HOCH', 'NOCH', 'SONNE', 'GUT', 'SCHOEN', 'MITTE', 'KLAR', 'SCHNEE', 'DANN', 'SONST', 'REGION', 'BESONDERS', 'WETTER', 'WEHEN', 'AB', 'DIENST', 'STEIGEN', 'IX', 'NACHT', 'ABEND', 'MEHR', 'TAG', 'HEUTE']}
#  bin class from CLIP b16
# binned_labels = {'bin_1': ['WO', 'DU', 'SELTEN', 'ICH', 'DAZWISCHEN', 'ALLE', 'UNTER', 'ALLGAEU', 'SECHSHUNDERT', 'VERAENDERN', 'KURZ', 'neg-HABEN', 'WINTER', 'negalp-AUCH', 'ZEIGEN-BILDSCHIRM', 'SEE', 'DURCHGEHEND', 'EIN-PAAR-TAGE', 'GEFRIEREN', 'TATSAECHLICH', 'MACHEN', 'VORAUSSAGE', 'MITTEILEN', 'M', 'UNTERSCHIED', 'ACHTE'], 'bin_2': ['VORSICHT', 'ERSTE', 'DRITTE', 'SKANDINAVIEN', 'ZONE', 'NULL', 'DAUER', 'WAS', 'GERADE', 'METER', 'GLEICH', 'WENIGER', 'HABEN', 'TAGSUEBER', 'WIE-IMMER', 'DARUM', 'GRAUPEL', 'neg-REGEN', 'NOVEMBER', 'SIEBEN', 'FUENFTE', 'ABWECHSELN', 'SCHOTTLAND', 'SINKEN', 'WIEDER', 'NACHMITTAG'], 'bin_3': ['SCHEINEN', 'WECHSEL', 'DREI', 'ZWEI', 'VORAUS', 'DIENSTAG', 'NAH', 'INFORMIEREN', 'TSCHUESS', 'JUNI', 'MITTWOCH', 'MAI', 'HOEHE', 'HABEN2', 'KOELN', 'GRUND', 'WIE-AUSSEHEN', 'REST', 'VOGEL', 'PLUS', 'ZWEITE', 'VIERZEHN', 'HAGEL', 'AUFLOESEN', 'GLATT', 'BURG'], 'bin_4': ['BAYERN', 'NAECHSTE', 'APRIL', 'FUENFZEHN', 'SECHSZEHN', 'TAUEN', 'DREISSIG', 'ZUERST', 'WOCHE', 'LOCKER', 'OFT', 'FUENF', 'LANGSAM', 'negalp-KEIN', 'ZWISCHEN', 'WEITER', 'WENIG', 'DREIZEHN', 'EINS', 'UEBER', 'DRUCK', 'ACHT', 'BRAND', 'SCHWER', 'IN-KOMMEND', 'MAXIMAL'], 'bin_5': ['MONTAG', 'WENN', 'SPAETER', 'HIMMEL', 'DOCH', 'SACHSEN', 'FEBRUAR', 'SUEDOST', 'DABEI', 'SUEDWEST', 'HERBST', 'STRASSE', 'SONNTAG', 'ZWOELF', 'SECHS', 'TRUEB', 'WALD', 'UND', 'ORT', 'SIEBZEHN', 'VERSCHIEDEN', 'BERLIN', 'NEUNZEHN', 'WUENSCHEN', 'DEUTSCHLAND', 'VOR'], 'bin_6': ['SAMSTAG', 'BODEN', 'LEICHT', 'AUGUST', 'MITTAG', 'ACHTZEHN', 'ZWANZIG', 'TEMPERATUR', 'AEHNLICH', 'TEIL', 'MAERZ', 'poss-EUCH', 'ZUSCHAUER', 'DAZU', 'BEWOELKT', 'ELF', 'FLUSS', 'BLEIBEN', 'FRUEH', 'SEPTEMBER', 'NEBEL', 'LAND', 'WIE', 'ANFANG', 'WAHRSCHEINLICH', 'NEUN'], 'bin_7': ['KALT', 'UNWETTER', 'EUROPA', 'BESSER', 'HAUPTSAECHLICH', 'SCHNEIEN', 'MIT', 'DESHALB', 'SUED', 'OKTOBER', 'MISCHUNG', 'STERN', 'UEBERWIEGEND', 'DEZEMBER', 'SPEZIELL', 'MAL', 'MINUS', 'VIER', 'SO', 'GRAD', 'ABER', 'DONNERSTAG', 'BEGRUESSEN', 'RUHIG', 'BIS', 'WECHSELHAFT'], 'bin_8': ['SEHEN', 'HOCH', 'STURM', 'LIEB', 'ZEHN', 'KUEHL', 'NUR', 'LUFT', 'FEUCHT', 'LANG', 'HEISS', 'MEISTENS', 'KOENNEN', 'WOCHENENDE', 'AB', 'GEFAHR', 'KOMMEN', 'SONST', 'NACH', 'IM-VERLAUF', 'MEER', 'GEWITTER', 'ODER', 'FRISCH', 'FREITAG', 'VERSCHWINDEN'], 'bin_9': ['TIEF', 'STARK', 'SCHWACH', 'GUT', 'WEST', 'NORDOST', 'ENORM', 'NEU', 'SCHAUER', 'OST', 'MORGEN', 'REGEN', 'BISSCHEN', 'ALPEN', 'JANUAR', 'VIEL', 'MILD', 'NORDWEST', 'BERG', 'DEUTSCH', 'FROST', 'MAESSIG', 'WEHEN', 'SCHON', 'AUCH', 'WARM'], 'bin_10': ['KUESTE', 'ORKAN', 'REGION', 'JETZT', 'MOEGLICH', 'WIND', 'WOLKE', 'WETTER', 'SCHNEE', 'SONNE', 'BESONDERS', 'NORD', 'WARNUNG', 'TEILWEISE', 'TAG', 'MITTE', 'KLAR', 'NOCH', 'IX', 'NACHT', 'JULI', 'STEIGEN', 'HEUTE', 'TROCKEN', 'FREUNDLICH', 'MEHR', 'SCHOEN', 'DIENST', 'ABEND', 'DANN']}
# bin from SignCL
# binned_labels = {'bin_1': ['WO', 'DU', 'negalp-AUCH', 'ICH', 'DAZWISCHEN', 'SELTEN', 'VERAENDERN', 'SECHSHUNDERT', 'UNTER', 'ALLE', 'DURCHGEHEND', 'ZEIGEN-BILDSCHIRM', 'ALLGAEU', 'neg-HABEN', 'GERADE', 'EIN-PAAR-TAGE', 'FEBRUAR', 'KURZ', 'SKANDINAVIEN', 'WIE-IMMER', 'GEFRIEREN', 'SEE', 'METER', 'SINKEN', 'TATSAECHLICH', 'M'], 'bin_2': ['WAS', 'MAI', 'MITTEILEN', 'INFORMIEREN', 'TAGSUEBER', 'ACHTE', 'SCHEINEN', 'HABEN', 'NACHMITTAG', 'DAUER', 'GLEICH', 'MACHEN', 'WINTER', 'ZONE', 'ERSTE', 'VORAUSSAGE', 'VORAUS', 'UNTERSCHIED', 'WENIGER', 'JUNI', 'MAERZ', 'GRAUPEL', 'neg-REGEN', 'SPAETER', 'DARUM', 'SIEBEN'], 'bin_3': ['FUENFTE', 'WIE-AUSSEHEN', 'AUGUST', 'DREI', 'DRITTE', 'DIENSTAG', 'APRIL', 'NULL', 'HOEHE', 'TSCHUESS', 'ZWEI', 'BEWOELKT', 'negalp-KEIN', 'JULI', 'AUFLOESEN', 'WECHSEL', 'NAH', 'SCHOTTLAND', 'DEZEMBER', 'NOVEMBER', 'BURG', 'REST', 'LANGSAM', 'PLUS', 'MITTWOCH', 'BEGRUESSEN'], 'bin_4': ['WUENSCHEN', 'TAUEN', 'WENIG', 'GRUND', 'BODEN', 'WIEDER', 'DABEI', 'OKTOBER', 'ZWISCHEN', 'WEITER', 'MAXIMAL', 'VERSCHIEDEN', 'VORSICHT', 'ABWECHSELN', 'DRUCK', 'IN-KOMMEND', 'WOCHE', 'GLATT', 'HABEN2', 'KOMMEN', 'DREIZEHN', 'EINS', 'BAYERN', 'ZUSCHAUER', 'NAECHSTE', 'HAGEL'], 'bin_5': ['ZUERST', 'HIMMEL', 'VOGEL', 'ACHT', 'SEHEN', 'NEBEL', 'SECHSZEHN', 'VIERZEHN', 'DREISSIG', 'FUENFZEHN', 'WAHRSCHEINLICH', 'BERLIN', 'UND', 'SACHSEN', 'HERBST', 'ANFANG', 'SUEDOST', 'SECHS', 'KOELN', 'DESHALB', 'LOCKER', 'DAZU', 'MEISTENS', 'GRAD', 'SO', 'ABER'], 'bin_6': ['VERSCHWINDEN', 'UEBER', 'ZWOELF', 'LIEB', 'TEIL', 'MITTAG', 'OFT', 'SUEDWEST', 'BRAND', 'WENN', 'WALD', 'KOENNEN', 'DEUTSCHLAND', 'ORT', 'SAMSTAG', 'SEPTEMBER', 'poss-EUCH', 'BLEIBEN', 'STERN', 'SUED', 'BESSER', 'GUT', 'MONTAG', 'VIER', 'ORKAN', 'NACH'], 'bin_7': ['FUENF', 'TRUEB', 'ZWEITE', 'FLUSS', 'NEUNZEHN', 'BISSCHEN', 'HAUPTSAECHLICH', 'WIE', 'MINUS', 'RUHIG', 'MISCHUNG', 'SONNTAG', 'ZEHN', 'TEMPERATUR', 'LAND', 'JETZT', 'SCHNEIEN', 'UEBERWIEGEND', 'LANG', 'NORDOST', 'KALT', 'ACHTZEHN', 'JANUAR', 'HOCH', 'KUEHL', 'SIEBZEHN'], 'bin_8': ['ALPEN', 'ZWANZIG', 'NUR', 'MIT', 'MAL', 'WECHSELHAFT', 'SCHWER', 'BERG', 'SCHON', 'MORGEN', 'AEHNLICH', 'LEICHT', 'MOEGLICH', 'STRASSE', 'WEHEN', 'WOLKE', 'BIS', 'STURM', 'FRISCH', 'ELF', 'GEFAHR', 'LUFT', 'WOCHENENDE', 'REGION', 'REGEN', 'SONST'], 'bin_9': ['OST', 'VOR', 'FRUEH', 'MILD', 'TIEF', 'FROST', 'EUROPA', 'KUESTE', 'SONNE', 'NORDWEST', 'DEUTSCH', 'STARK', 'GEWITTER', 'FREITAG', 'SCHWACH', 'DOCH', 'WEST', 'SCHAUER', 'DONNERSTAG', 'MEER', 'FEUCHT', 'BESONDERS', 'SPEZIELL', 'MITTE', 'NORD', 'UNWETTER'], 'bin_10': ['AB', 'HEISS', 'AUCH', 'WIND', 'WARM', 'IM-VERLAUF', 'NEUN', 'IX', 'KLAR', 'NEU', 'WETTER', 'WARNUNG', 'ODER', 'VIEL', 'TAG', 'ENORM', 'SCHNEE', 'MEHR', 'MAESSIG', 'HEUTE', 'DANN', 'TROCKEN', 'NACHT', 'ABEND', 'SCHOEN', 'NOCH', 'DIENST', 'FREUNDLICH', 'TEILWEISE', 'STEIGEN']}
# Load the JSON file
# sdr_path = "/home/chan0305/sign_lang/Sign-IR/examples/GFSLT-VLP/out/SignIR_baseline"
sdr_path = "/home/chan0305/sign_lang/Sign-IR/examples/GFSLT-VLP/out/SignCL_eval"
# with open(f"{sdr_path}/sdr_values_1.json", 'r') as file:
#     sdr_values1 = json.load(file)

with open(f"{sdr_path}/sdr_values_2.json", 'r') as file:
    sdr_values2 = json.load(file)

# sdr_values = {**sdr_values1, **sdr_values2}
sdr_values = sdr_values2
# Filter out entries with a value of 0
filtered_items = [(label, value) for label, value in sdr_values.items() if value != 0]

# Sort the filtered SDR values
sorted_items = sorted(filtered_items, key=lambda item: item[1])

# Define the number of bins
num_bins = 10

# Calculate the number of elements per bin
elements_per_bin = len(sorted_items) // num_bins

# Initialize bins
if binned_labels is None:
    binned_labels = {f'bin_{i+1}': [] for i in range(num_bins)}
    bin_averages = {f'bin_{i+1}': 0 for i in range(num_bins)}
    bin_variances = {f'bin_{i+1}': 0 for i in range(num_bins)}
    # Separate labels into bins and calculate the average value for each bin
    for i in range(num_bins):
        start_index = i * elements_per_bin
        if i == num_bins - 1:  # Last bin takes the remaining elements
            end_index = len(sorted_items)
        else:
            end_index = (i + 1) * elements_per_bin
        bin_values = [value for label, value in sorted_items[start_index:end_index]]
        bin_averages[f'bin_{i+1}'] = np.mean(bin_values)
        bin_variances[f'bin_{i+1}'] = np.var(bin_values)
        for label, value in sorted_items[start_index:end_index]:
            binned_labels[f'bin_{i+1}'].append(label)
else:
    bin_averages = {f'bin_{i+1}': 0 for i in range(num_bins)}
    bin_variances = {f'bin_{i+1}': 0 for i in range(num_bins)}
    for key, val in binned_labels.items():
        bin_values = []
        for label in val:
            bin_values.append(sdr_values[label])
        bin_averages[key] = np.mean(bin_values)
        bin_variances[key] = np.var(bin_values)
# print(binned_labels)
# # Print the results
for bin_label, labels in binned_labels.items():
    # print(f"{bin_label}: {sorted(labels)}")
    print(f"Average value in {bin_label}: {bin_averages[bin_label]}")
    print(f"Variance in {bin_label}: {bin_variances[bin_label]}")
total_average = np.mean([value for label, value in sorted_items])
print(f"Total average value: {total_average}")
with open(f"{sdr_path}/frame_features_1.pickle", 'rb') as file:
    feature_dict1 = pickle.load(file)

with open(f"{sdr_path}/frame_features_2.pickle", 'rb') as file:
    feature_dict2 = pickle.load(file)
# feature_dict = {**feature_dict1, **feature_dict2}


# feature_dict = feature_dict2
# # save_dir = "./out/SignIR_noSignCL_eval"  # Replace with the actual save directory
# tsne_visualize_bins(feature_dict, binned_labels, sdr_path)