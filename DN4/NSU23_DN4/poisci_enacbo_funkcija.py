import numpy as np
import pandas as pd
import ProGED as pg
from collections import defaultdict
from nltk.grammar import ProbabilisticProduction
from nltk.grammar import PCFG
from nltk import Nonterminal

def flatten_rules(data):
    """
    sprejme drevesno obliko poti iz gramatike
    v obliki seznama seznamov in vrne en seznam
    """
    rules = []

    def helper(nested_rules):
        for rule in nested_rules:
            if isinstance(rule, list):
                helper(rule)
            else:
                rules.append(rule)

    for key in data:
        helper(data[key][1])

    return rules


def gramatika_v_slovar(gramatika):
    """
    sprejme seznam generiran s funkcijo flatten_rules()
    in vrne seznam v obliki slovarja: pravilo -> vrednosti [verjetnosti]
    """
    slovar = defaultdict(dict)
    for element in gramatika:
        slovar[element.lhs()][element.rhs()] = element.prob()
    return slovar


def posodobi_gramatiko(prvotni_slovar, enacba_slovar, is_error_big, sprememba=0.01):
    """
    sprejme 2 slovarja, enega s celotno gramatiko, enega z gramatiko poti enačbe
    sprejme tudi logično vrednost ali je napaka velika ali majhna
    posodablja celotno gramatiko po pravilu:
        - če je bilo pravilo uporabljeno za odkrivanje enačbe in je napaka velika zmanjša njegovo verjetnost
        - če je bilo pravilo uporabljeno za odkrivanje enačbe in napaka ni velika poveča njegovo verjetnost
        - če pravilo ni bilo uporabljeno za odkrivanje enačbe verjetnost ostane enaka
    standardizira verjetnosti
    """
    enacba = {item for sublist in map(list, enacba_slovar.values()) for item in sublist}

    for var, rules in prvotni_slovar.items():
        num = len(enacba_slovar[var])
        s = 0

        for rule, prob in rules.items():
            if rule in enacba:
                if is_error_big:
                    prvotni_slovar[var][rule] = max(prob - num * sprememba, 0)
                else:
                    prvotni_slovar[var][rule] += num * sprememba

            s += prvotni_slovar[var][rule]

        if s:  # Avoid division by zero
            for rule in rules:
                prvotni_slovar[var][rule] /= s

    return prvotni_slovar


def ustvari_gramatiko(slovar):
    """
    pretvori slovar nazaj v gramatiko primerno za funkcijo Eq.disco
    """
    gramatika = [ProbabilisticProduction(var, list(rule), prob=prob)
                 for var, dict_rull_prob in slovar.items()
                 for rule, prob in dict_rull_prob.items()]
    
    return pg.GeneratorGrammar(PCFG(Nonterminal('E'), gramatika))


def poisci_enacbo(gramatika, podatki, stevilo_iteracij = 100, sprememba=0.01, rhs_vars = ["x1", "x2", "x3", "x4", "x5"], eps=10):
    """
    glavna funkcija, uporabi vse zgornje, da preko while zanke
    odkrije pravo enačbo s posodabljanjem gramatike
    """
    grammar = pg.GeneratorGrammar(gramatika)
    ED = pg.EqDisco(data=podatki, 
                    sample_size=stevilo_iteracij,
                    lhs_vars=["y"],
                    rhs_vars=rhs_vars,
                    verbosity=1,
                    generator = grammar)
    ED.generate_models()
    model = ED.fit_models()
    rezultati = ED.get_results()

    napaka = rezultati[0].get_error()
    min_error = napaka

    prvotni_slovar = gramatika_v_slovar(grammar.grammar.productions())
    enacba_slovar = gramatika_v_slovar(flatten_rules(model[0].info['trees']))

    n = 0
    napaka_nova = 10
    while n < stevilo_iteracij and napaka_nova >= eps:
        nov_slovar = posodobi_gramatiko(prvotni_slovar, enacba_slovar, napaka_nova > 10 or napaka < napaka_nova)
        
        nova_gramatika = ustvari_gramatiko(nov_slovar)
        print(nova_gramatika)
        ED2 = pg.EqDisco(data=podatki, 
                sample_size=1,
                lhs_vars=["y"],
                rhs_vars=rhs_vars,
                verbosity=1,
                generator = nova_gramatika)
        ED2.generate_models()
        model2 = ED2.fit_models()
        rezultati2 = ED2.get_results(3)

        prvotni_slovar = gramatika_v_slovar(nova_gramatika.grammar.productions())
        enacba_slovar = gramatika_v_slovar(flatten_rules(model2[0].info['trees']))

        napaka = napaka_nova
        napaka_nova = rezultati2[0].get_error()
        min_error = min(min_error, napaka_nova)
        n += 1

    print(nova_gramatika)
    print(rezultati2)
    return([rezultati2, min_error])