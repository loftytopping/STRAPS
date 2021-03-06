##########################################################################################
#											 #
#    Keys descriptors associated with EI spectra predictions                             # 
#                                                                                        #
#                                                                                        #
#    Copyright (C) 2016  David Topping : david.topping@manchester.ac.uk                  #
#                                      : davetopp80@gmail.com                            # 
#    Personal website: davetoppingsci.com                                                #
#                                                                                        # 
#    This program is free software: you can redistribute it and/or modify                #
#    it under the terms of the GNU Affero General Public License as published            #
#    by the Free Software Foundation, either version 3 of the License, or                #
#    (at your option) any later version.                                                 # 
#                                                                                        #   
#    This program is distributed in the hope that it will be useful,                     #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of                      # 
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                       #
#    GNU Affero General Public License for more details.                                 #
#                                                                                        #
#    You should have received a copy of the GNU Affero General Public License            # 
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.               #
#                                                                                        #
#                                                                                        #
#                                                                                        # 
#                                                                                        #
##########################################################################################


# NOTE: David Topping
# I have created this file by extracting the MACCS and FP4 features used in the parsing routine
# of the RDKit package
#
# For the MACCS - SMARTS KEYS
# Copyright (C) 2001-2011 greg Landrum and Rational Discovery LLC
#
""" SMARTS definitions for the publically available MACCS keys
and a MACCS fingerprinter

I compared the MACCS fingerprints generated here with those from two
other packages (not MDL, unfortunately). Of course there are
disagreements between the various fingerprints still, but I think
these definitions work pretty well. Some notes:

1) most of the differences have to do with aromaticity
2) there's a discrepancy sometimes because the current RDKit
definitions do not require multiple matches to be distinct. e.g. the
SMILES C(=O)CC(=O) can match the (hypothetical) key O=CC twice in my
definition. It's not clear to me what the correct behavior is.
3) Some keys are not fully defined in the MDL documentation
4) Two keys, 125 and 166, have to be done outside of SMARTS.
5) Key 1 (ISOTOPE) isn't defined

Rev history:
2006 (gl): Original open-source release
May 2011 (gl): Update some definitions based on feedback from Andrew Dalke

"""
def MACCS_tags():

    # these are SMARTS patterns corresponding to the MDL MACCS keys
    smartsPatts={
    1:'#ISOTOPENotcomplete',
    2:'#GroupIVa,Va,VIaPeriods04/06/16(Ge...)*NOTE*specwrong',
    3:'#actinide',
    4:'#GroupIIIB,IVB(Sc...)*NOTE*specwrong',
    5:'#Lanthanide',
    6:'#GroupVB,VIB,VIIB(V...)*NOTE*specwrong',
    7:'#QAAA@1',
    8:'#GroupVIII(Fe...)',
    9:'#GroupIIa(Alkalineearth)',
    10:'#4MRing',
    11:'#GroupIB,IIB(Cu..)',
    12:'#ON(C)C',
    13:'#S-S',
    14:'#OC(O)O',
    15:'#QAA@1',
    16:'#CTC',
    17:'#GroupIIIA(B...)*NOTE*specwrong',
    18:'#7MRing',
    19:'#Si',
    20:'#C=C(Q)Q',
    21:'#3MRing',
    22:'#NC(O)O',
    23:'#N-O',
    24:'#NC(N)N',
    25:'#C$=C($A)$A',
    26:'#I',
    27:'#QCH2Q',
    28:'#P',
    29:'#CQ(C)(C)A',
    30:'#QX',
    31:'#CSN',
    32:'#NS',
    33:'#CH2=A',
    34:'#GroupIA(AlkaliMetal)',
    35:'#SHeterocycle',
    36:'#NC(O)N',
    37:'#NC(C)N',
    38:'#OS(O)O',
    39:'#S-O',
    40:'#CTN',
    41:'#F',
    42:'#QHAQH',
    43:'#C=CN',
    44:'#BR',
    45:'#SAN',
    46:'#OQ(O)O',
    47:'#CHARGE',
    48:'#C=C(C)C',
    49:'#CSO',
    50:'#NN',
    51:'#QHAAAQH',
    52:'#QHAAQH',
    53:'#OSO',
    54:'#ON(O)C',
    55:'#OHeterocycle',
    56:'#QSQ',
    57:'#Snot%A%A',
    58:'#S=O',
    59:'#AS(A)A',
    60:'#A$!A$A',
    61:'#N=O',
    62:'#A$A!S',
    63:'#C%N',
    64:'#CC(C)(C)A',
    65:'#QS',
    66:'#QHQH(&...)FIX:incompletedefinition',
    67:'#QQH',
    68:'#QNQ',
    69:'#NO',
    70:'#OAAO',
    71:'#S=A',
    72:'#CH3ACH3',
    73:'#A!N$A',
    74:'#C=C(A)A',
    75:'#NAN',
    76:'#C=N',
    77:'#NAAN',
    78:'#NAAAN',
    79:'#SA(A)A',
    80:'#ACH2QH',
    81:'#QAAAA@1',
    82:'#NH2',
    83:'#CN(C)C',
    84:'#CH2QCH2',
    85:'#X!A$A',
    86:'#S',
    87:'#OAAAO',
    88:'#QHAACH2A',
    89:'#QHAAACH2A',
    90:'#OC(N)C',
    91:'#QCH3',
    92:'#QN',
    93:'#NAAO',
    94:'#5Mring',
    95:'#NAAAO',
    96:'#QAAAAA@1',
    97:'#C=C',
    98:'#ACH2N',
    99:'#8MRingorlarger.Thisonlyhandlesupto',
    100:'#QO',
    101:'#CL',
    102:'#QHACH2A',
    103:'#A$A($A)$A',
    104:'#QA(Q)Q',
    105:'#XA(A)A',
    106:'#CH3AAACH2A',
    107:'#ACH2O',
    108:'#NCO',
    109:'#NACH2A',
    110:'#AA(A)(A)A',
    111:'#Onot%A%A',
    112:'#CH3CH2A',
    113:'#CH3ACH2A',
    114:'#CH3AACH2A',
    115:'#NAO',
    116:'#ACH2CH2A>1',
    117:'#N=A',
    118:'#Heterocyclicatom>1(&...)FIX:incompletedefinition',
    119:'#NHeterocycle',
    120:'#AN(A)A',
    121:'#OCO',
    122:'#QQ',
    123:'#AromaticRing>1',
    124:'#A!O!A',
    125:'#A$A!O>1(&...)FIX:incompletedefinition',
    126:'#ACH2AAACH2A',
    127:'#ACH2AACH2A',
    128:'#QQ>1(&...)FIX:incompletedefinition',
    129:'#QH>1',
    130:'#OACH2A',
    131:'#A$A!N',
    132:'#X(HALOGEN)',
    133:'#Nnot%A%A',
    134:'#O=A>1',
    135:'#Heterocycle',
    136:'#QCH2A>1(&...)FIX:incompletedefinition',
    137:'#OH',
    138:'#O>3(&...)FIX:incompletedefinition',
    139:'#CH3>2(&...)FIX:incompletedefinition',
    140:'#N>1',
    141:'#A$A!O',
    142:'#Anot%A%Anot%A',
    143:'#6Mring>1',
    144:'#O>2',
    145:'#ACH2CH2A',
    146:'#AQ(A)A',
    147:'#CH3>1',
    148:'#A!A$A!A',
    149:'#NH',
    150:'#OC(C)C',
    151:'#QCH2A',
    152:'#C=O',
    153:'#A!CH2!A',
    154:'#NA(A)A',
    155:'#C-O',
    156:'#C-N',
    157:'#O>1',
    158:'#CH3',
    159:'#N',
    160:'#Aromatic',
    161:'#6MRing',
    162:'#O',
    163:'#Ring'
    }
    
    return smartsPatts

# 2 FP4 keys
#
#              SMARTS Patterns for Functional Group Classification 
#
#              written by Christian Laggner 
#              Copyright 2005 Inte:Ligand Software-Entwicklungs und Consulting GmbH
#
#              Released under the Lesser General Public License (LGPL license)
#              see http://www.gnu.org/copyleft/lesser.html
#              Modified from Version 221105
#####################################################################################################
# General Stuff:
# These patterns were written in an attempt to represent the classification of organic compounds 
# from the viewpoint of an organic chemist.
# They are often very restrictive. This may be generally a good thing, but it also takes some time
# for filtering/indexing large compound sets. 
# For filtering undesired groups (in druglike compounds) one will want to have more general patterns 
# (e.g. you don't want *any* halide of *any* acid, *neither* aldehyde *nor* formyl esters and amides, ...). 

def FP4_tags():    

    smartsPatts={
    
    1:'Primary_carbon:',
    2:'Secondary_carbon:',
    3:'Tertiary_carbon:',
    4:'Quaternary_carbon:',
    5:'Alkene:',
    6:'Alkyne:',
    7:'Allene:',
    8:'Alkylchloride:',
    9:'Alkylfluoride:',
    10:'Alkylbromide:',
    11:'Alkyliodide:',
    12:'Alcohol:',
    13:'Primary_alcohol:',
    14:'Secondary_alcohol:',
    15:'Tertiary_alcohol:',
    16:'Dialkylether:',
    17:'Dialkylthioether:',
    18:'Alkylarylether:',
    19:'Diarylether:',
    20:'Alkylarylthioether:',
    21:'Diarylthioether:',
    22:'Oxonium:',
    23:'Amine:',
    24:'Primary_aliph_amine:',
    25:'Secondary_aliph_amine:',
    26:'Tertiary_aliph_amine:',
    27:'Quaternary_aliph_ammonium:',
    28:'Primary_arom_amine:',
    29:'Secondary_arom_amine:',
    30:'Tertiary_arom_amine:',
    31:'Quaternary_arom_ammonium:',
    32:'Secondary_mixed_amine:',
    33:'Tertiary_mixed_amine:',
    34:'Quaternary_mixed_ammonium:',
    35:'Ammonium:',
    36:'Alkylthiol:',
    37:'Dialkylthioether:',
    38:'Alkylarylthioether:',
    39:'Disulfide:',
    40:'1,2-Aminoalcohol:',
    41:'1,2-Diol:',
    42:'1,1-Diol:',
    43:'Hydroperoxide:',
    44:'Peroxo:',
    45:'Organolithium_compounds:',
    46:'Organomagnesium_compounds:',
    47:'Organometallic_compounds:',
    48:'Aldehyde:',
    49:'Ketone:',
    50:'Thioaldehyde:',
    51:'Thioketone:',
    52:'Imine:',
    53:'Immonium:',
    54:'Oxime:',
    55:'Oximether:',
    56:'Acetal:',
    57:'Hemiacetal:',
    58:'Aminal:',
    59:'Hemiaminal:',
    60:'Thioacetal:',
    61:'Thiohemiacetal:',
    62:'Halogen_acetal_like:',
    63:'Acetal_like:',
    64:'Halogenmethylen_ester_and_similar:',
    65:'NOS_methylen_ester_and_similar:',
    66:'Hetero_methylen_ester_and_similar:',
    67:'Cyanhydrine:',
    68:'Chloroalkene:',
    69:'Fluoroalkene:',
    70:'Bromoalkene:',
    71:'Iodoalkene:',
    72:'Enol:',
    73:'Endiol:',
    74:'Enolether:',
    75:'Enolester:',
    76:'Enamine:',
    77:'Thioenol:',
    78:'Thioenolether:',
    79:'Acylchloride:',
    80:'Acylfluoride:',
    81:'Acylbromide:',
    82:'Acyliodide:',
    83:'Acylhalide:',
    84:'Carboxylic_acid:',
    85:'Carboxylic_ester:',
    86:'Lactone:',
    87:'Carboxylic_anhydride:',
    88:'Carboxylic_acid_derivative:',
    89:'Carbothioic_acid:',
    90:'Carbothioic_S_ester:',
    91:'Carbothioic_S_lactone:',
    92:'Carbothioic_O_ester:',
    93:'Carbothioic_O_lactone:',
    94:'Carbothioic_halide:',
    95:'Carbodithioic_acid:',
    96:'Carbodithioic_ester:',
    97:'Carbodithiolactone:',
    98:'Amide:',
    99:'Primary_amide:',
    100:'Secondary_amide:',
    101:'Tertiary_amide:',
    102:'Lactam:',
    103:'Alkyl_imide:',
    104:'N_hetero_imide:',
    105:'Imide_acidic:',
    106:'Thioamide:',
    107:'Thiolactam:',
    108:'Oximester:',
    109:'Amidine:',
    110:'Hydroxamic_acid:',
    111:'Hydroxamic_acid_ester:',
    112:'Imidoacid:',
    113:'Imidoacid_cyclic:',
    114:'Imidoester:',
    115:'Imidolactone:',
    116:'Imidothioacid:',
    117:'Imidothioacid_cyclic:',
    118:'Imidothioester:',
    119:'Imidothiolactone:',
    120:'Amidine:',
    121:'Imidolactam:',
    122:'Imidoylhalide:',
    123:'Imidoylhalide_cyclic:',
    125:'Amidrazone:',
    126:'Alpha_aminoacid:',
    127:'Alpha_hydroxyacid:',
    128:'Peptide_middle:',
    129:'Peptide_C_term:',
    130:'Peptide_N_term:',
    131:'Carboxylic_orthoester:',
    132:'Ketene:',
    133:'Ketenacetal:',
    134:'Nitrile:',
    135:'Isonitrile:',
    136:'Vinylogous_carbonyl_or_carboxyl_derivative:',
    137:'Vinylogous_acid:',
    138:'Vinylogous_ester:',
    139:'Vinylogous_amide:',
    140:'Vinylogous_halide:',
    141:'Carbonic_acid_dieester:',
    142:'Carbonic_acid_esterhalide:',
    143:'Carbonic_acid_monoester:',
    144:'Carbonic_acid_derivatives:',
    145:'Thiocarbonic_acid_dieester:',
    146:'Thiocarbonic_acid_esterhalide:',
    147:'Thiocarbonic_acid_monoester:',
    148:'Urea:[#7X3;!$([#7][!#6])][#6X3](=[OX1])[#7X3;!$([#7][!#6])]',
    149:'Thiourea:',
    150:'Isourea:',
    151:'Isothiourea:',
    152:'Guanidine:',
    153:'Carbaminic_acid:',
    154:'Urethan:',
    155:'Biuret:',
    156:'Semicarbazide:',
    157:'Carbazide:',
    158:'Semicarbazone:',
    159:'Carbazone:',
    160:'Thiosemicarbazide:',
    161:'Thiocarbazide:',
    162:'Thiosemicarbazone:',
    163:'Thiocarbazone:',
    164:'Isocyanate:',
    165:'Cyanate:',
    166:'Isothiocyanate:',
    167:'Thiocyanate:',
    168:'Carbodiimide:',
    169:'Orthocarbonic_derivatives:',
    170:'Phenol:',
    171:'1,2-Diphenol:',
    172:'Arylchloride:',
    173:'Arylfluoride:',
    174:'Arylbromide:',
    175:'Aryliodide:',
    176:'Arylthiol:',
    177:'Iminoarene:',
    178:'Oxoarene:',
    179:'Thioarene:',
    180:'Hetero_N_basic_H:',
    181:'Hetero_N_basic_no_H:',
    182:'Hetero_N_nonbasic:',
    183:'Hetero_O:',
    184:'Hetero_S:',
    185:'Heteroaromatic:',
    186:'Nitrite:',
    187:'Thionitrite:',
    188:'Nitrate:',
    189:'Nitro:',
    190:'Nitroso:',
    191:'Azide:',
    192:'Acylazide:',
    193:'Diazo:',
    194:'Diazonium:',
    195:'Nitrosamine:',
    196:'Nitrosamide:',
    197:'N-Oxide:',
    198:'Hydrazine:',
    199:'Hydrazone:',
    200:'Hydroxylamine:',
    201:'Sulfon:',
    202:'Sulfoxide:',
    203:'Sulfonium:',
    204:'Sulfuric_acid:',
    205:'Sulfuric_monoester:',
    206:'Sulfuric_diester:',
    207:'Sulfuric_monoamide:',
    208:'Sulfuric_diamide:',
    209:'Sulfuric_esteramide:',
    210:'Sulfuric_derivative:',
    211:'Sulfonic_acid:',
    212:'Sulfonamide:',
    213:'Sulfonic_ester:',
    214:'Sulfonic_halide:',
    215:'Sulfonic_derivative:',
    216:'Sulfinic_acid:',
    217:'Sulfinic_amide:',
    218:'Sulfinic_ester:',
    219:'Sulfinic_halide:',
    220:'Sulfinic_derivative:',
    221:'Sulfenic_acid:',
    222:'Sulfenic_amide:',
    223:'Sulfenic_ester:',
    224:'Sulfenic_halide:',
    225:'Sulfenic_derivative:',
    226:'Phosphine:',
    227:'Phosphine_oxide:',
    228:'Phosphonium:',
    229:'Phosphorylen:',
    230:'Phosphonic_acid:',
    231:'Phosphonic_monoester:',
    232:'Phosphonic_diester:',
    233:'Phosphonic_monoamide:',
    234:'Phosphonic_diamide:',
    235:'Phosphonic_esteramide:',
    236:'Phosphonic_acid_derivative:',
    237:'Phosphoric_acid:',
    238:'Phosphoric_monoester:',
    239:'Phosphoric_diester:',
    240:'Phosphoric_triester:',
    241:'Phosphoric_monoamide:',
    242:'Phosphoric_diamide:',
    243:'Phosphoric_triamide:',
    244:'Phosphoric_monoestermonoamide:',
    245:'Phosphoric_diestermonoamide:',
    246:'Phosphoric_monoesterdiamide:',
    247:'Phosphoric_acid_derivative:',
    248:'Phosphinic_acid:',
    249:'Phosphinic_ester:',
    250:'Phosphinic_amide:',
    251:'Phosphinic_acid_derivative:',
    252:'Phosphonous_acid:',
    253:'Phosphonous_monoester:',
    254:'Phosphonous_diester:',
    255:'Phosphonous_monoamide:',
    256:'Phosphonous_diamide:',
    257:'Phosphonous_esteramide:',
    258:'Phosphonous_derivatives:',
    259:'Phosphinous_acid:',
    260:'Phosphinous_ester:',
    261:'Phosphinous_amide:',
    262:'Phosphinous_derivatives:',
    263:'Quart_silane:',
    264:'Non-quart_silane:',
    265:'Silylmonohalide:',
    266:'Het_trialkylsilane:',
    267:'Dihet_dialkylsilane:',
    268:'Trihet_alkylsilane:',
    269:'Silicic_acid_derivative:',
    270:'Trialkylborane:',
    271:'Boric_acid_derivatives:',
    272:'Boronic_acid_derivative:',
    273:'Borohydride:',
    274:'Quaternary_boron:',
    275:'Aromatic:',
    276:'Heterocyclic:',
    277:'Epoxide:',
    278:'NH_aziridine:',
    279:'Spiro:',
    280:'Annelated_rings:',
    281:'Bridged_rings:',
    282:'Sugar_pattern_1:',
    283:'Sugar_pattern_2:',
    284:'Sugar_pattern_combi:',
    285:'Sugar_pattern_2_reducing:',
    286:'Sugar_pattern_2_alpha:',
    287:'Sugar_pattern_2_beta:',
    288:'Conjugated_double_bond:',
    289:'Conjugated_tripple_bond:',
    290:'Cis_double_bond:',
    291:'Trans_double_bond:',
    292:'Mixed_anhydrides:',
    293:'Halogen_on_hetero:',
    294:'Halogen_multi_subst:',
    295:'Trifluoromethyl:',
    296:'C_ONS_bond:',
    297:'Charged:',
    298:'Anion:',
    299:'Kation:',
    302:'1,3-Tautomerizable:',
    303:'1,5-Tautomerizable:',
    304:'Rotatable_bond:',
    305:'Michael_acceptor:',
    306:'Dicarbodiazene:',
    307:'CH-acidic:',
    308:'CH-acidic_strong:',
    309:'Chiral_center_specified:'
    }
    
    return smartsPatts

# 2 Nanoolal Primary keys
# SMARTS keys were constructed by Dr Mark Barley, University of Manchester.

def Nanoolal_P_tags():    

    smartsPatts={
    1:'(CH3) attached to carbon',
    2:'(CH3) attached to electronegative (non aromatic) atom',
    3:'(CH3) attached to an aromatic atom',
    4:'(CH2) attached to a (C) in a chain',
    5:'(CH) attached to a (C) in a chain',
    6:'(>C<) attached to a (C) in a chain',
    7:'(CHx) in a chain attached to one or more electronegative atoms',
    8:'(CHx) attached to an aromatic atom',
    9:'(CH2) in a ring',
    10:'(>CH) in a ring',
    11:'(>C<) in a ring',
    12:'(CHx) in a ring attached to at least one electronegative atom that is not part of a ring',
    13:'(CHx) in a ring attached to at least one N or O that is part of a ring',
    14:'(CHx) in a ring attached to an aromatic atom',
    15:'Aromatic C-H',
    16:'C in an aromatic ring attached to aliphatic C',
    17:'C in an aromatic ring attached to electronegative atom',
    24:'F attached to an aromatic ring',
    25:'A single Cl attached to non-aromatic C',
    26:'Pairs of Cl attached to C',
    27:'Triplets of Cl attached to C',
    28:'Cl attached to aromatic C',
    29:'Cl attached to double bonded C',
    30:'Aliphatic bromides',
    31:'Aromatic bromides',
    32:'Iodides',
    33:'Tertiary alcohols',
    34:'Secondary alcohols',
    35:'Long chain (35) alcohols. ',
    36:'short chain (36) alcohols',
    37:'Aromatic alcohols or phenols',
    38:'Ethers (group 38 hits all ethers, while 65 hits aromatic ethers of the type found in Furans).',
    39:'Epoxides',
    40:'Aliphatic primary amines',
    41:'Aromatic primary amines',
    42:'Secondary amines: aliphatic or aromatic',
    43:'Tertiary amines: aliphatic or aromatic',
    44:'Carboxylic acids',
    45:'Ester group in a chain',
    46:'Formate group',
    47:'Ester within a ring-lactone',
    48:'Tertiary amide C(=O)N<',
    49:'Secondary amide C(=O)NH-',
    50:'Primary amide C(=O)NH2',
    51:'Aliphatic ketone',
    52:'Aliphatic aldehyde',
    57:'Nitrile group',
    58:'>C=C< in a chain with each C having at least one non-H neighbour',
    59:'???',
    60:'>C=C< in a chain attached to at least one electronegative atom',
    61:'CH2=C< in a chain',
    62:'>C=C< in a ring',
    63:'-C#C- in a chain',
    64:'HC#C- at end of chain',
    68:'Aliphatic nitro group',
    69:'Aromatic nitro group',
    72:'Nitrate group',
    76:'Acid anhydride',
    77:'Acid chloride',
    79:'Carbonate in a chain',
    88:'>C=C-C=C< in a ring',
    89:'>C=C-C=C< in a chain',
    90:'Aromatic aldehyde',
    92:'Aromatic ketone',
    94:'Bridging peroxide',
    97:'Secondary amine in a ring',
    103:'Cyclic carbonate',
    301:'Hydroperoxide group',
    302:'Peroxyacids',
    303:'PAN'
    }
    
    return smartsPatts


