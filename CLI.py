import argparse
import pandas as pd
import sys

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Run a specified algorithm on a given data file.')
    parser.add_argument('-a', '--algorithm', type=str, required=True,
                        help='Algorithm to run (e.g., O, S, C, SM, CM)')
    parser.add_argument('-l', '--limbs', type=int, required=True,
                        help='Number of limbs (e.g., 1-4)')
    parser.add_argument('-d', '--datafile', type=str, required=True,
                        help='Path to the data file (e.g., data_table.csv)')
    
    # ---- HMM (Markov) optional knobs; harmless for non-HMM runs ----
    parser.add_argument('--tau', type=float, default=1.0,
                        help='CK threshold pivot for HMM emissions (default: 1.0)')
    parser.add_argument('--alpha', type=float, default=6.0,
                        help='Logistic steepness for HMM emissions (default: 6.0)')
    parser.add_argument('--p_sw', type=float, default=0.008,
                        help='Per-epoch Sleep->Wake transition prob (default: 0.008)')
    parser.add_argument('--p_ws', type=float, default=0.025,
                        help='Per-epoch Wake->Sleep transition prob (default: 0.025)')
    parser.add_argument('--weights', type=str, default="",
                        help='Limb weights like "1:0.5,2:2.0,3:0.5,4:2.0" (HMM only)')
    
    args = parser.parse_args()
    
    # Load the data file
    try:
        df = pd.read_csv(args.datafile)
    except FileNotFoundError:
        sys.exit(f"Error: The data file '{args.datafile}' was not found.")
    except pd.errors.EmptyDataError:
        sys.exit(f"Error: The data file '{args.datafile}' is empty.")
    except Exception as e:
        sys.exit(f"Error reading '{args.datafile}': {e}")
    
    def parse_weights(weights_str, num_limbs):
        if not weights_str:
            return {i: 1.0 for i in range(1, num_limbs + 1)}
        out = {}
        for part in [p.strip() for p in weights_str.split(",") if p.strip()]:
            k, v = part.split(":")
            out[int(k.strip())] = float(v.strip())
        return out

    # Run the selected algorithm
    if args.algorithm == 'C':
        try:
            from apply_cole_kripke import apply_cole_kripke_single
        except ImportError:
            sys.exit("Error: Could not import 'apply_cole_kripke_single' from 'apply_cole_kripke.py'. "
                     "Ensure the file exists and is in the Python path.")
        result = apply_cole_kripke_single(df)
        
    elif args.algorithm == 'CM':
        try:
            from apply_cole_kripke import apply_cole_kripke_mult
        except ImportError:
            sys.exit("Error: Could not import 'apply_cole_kripke_mult' from 'apply_cole_kripke.py'. "
                     "Ensure the file exists and is in the Python path.")
        result = apply_cole_kripke_mult(df, args.limbs)

    elif args.algorithm == 'CMM':
        try:
            from apply_cole_kripke import apply_cole_kripke_mult_majority
        except ImportError:
            sys.exit("Error: Could not import 'apply_cole_kripke_mult_majority'.")
        result = apply_cole_kripke_mult_majority(df, args.limbs)

    elif args.algorithm == 'CW':
        try:
            from apply_cole_kripke import apply_cole_kripke_mult_weighted
        except ImportError:
            sys.exit("Error: Could not import 'apply_cole_kripke_mult_weighted'.")
        result = apply_cole_kripke_mult_weighted(df, args.limbs)

        # ---------- Minimal additions: HMM post-processing variants ----------
    elif args.algorithm == 'CH':
        # single-sensor Cole–Kripke + HMM (no new folders)
        try:
            from apply_cole_kripke import apply_cole_kripke_single
        except ImportError:
            sys.exit("Error: Could not import 'apply_cole_kripke_single'.")
        try:
            from hmm_postprocess import apply_hmm_over_cole
        except ImportError:
            sys.exit("Error: Could not import 'apply_hmm_over_cole' (hmm_postprocess.py missing?).")

        ck = apply_cole_kripke_single(df, output_file="cole_single_results.csv")
        # adapt single-sensor output to HMM input format (1 limb)
        tmp = pd.DataFrame({
            'dataTimestamp': ck['dataTimestamp'],
            'Limb 1 sleep_index': ck['sleep_index'],
            'Limb 1 sleep': ck['sleep'],
        })
        result = apply_hmm_over_cole(
            tmp, num_limbs=1, tau=args.tau, alpha=args.alpha,
            p_sw=args.p_sw, p_ws=args.p_ws, limb_weights={1: 1.0},
            add_probs=True, outfile="cole_single_hmm_results.csv"
        )

    elif args.algorithm == 'CMH':
        # multi-limb Cole–Kripke + HMM
        try:
            from apply_cole_kripke import apply_cole_kripke_mult
        except ImportError:
            sys.exit("Error: Could not import 'apply_cole_kripke_mult'.")
        try:
            from hmm_postprocess import apply_hmm_over_cole
        except ImportError:
            sys.exit("Error: Could not import 'apply_hmm_over_cole' (hmm_postprocess.py missing?).")

        ck = apply_cole_kripke_mult(df, args.limbs)
        limb_weights = parse_weights(args.weights, args.limbs)
        result = apply_hmm_over_cole(
            ck, num_limbs=args.limbs, tau=args.tau, alpha=args.alpha,
            p_sw=args.p_sw, p_ws=args.p_ws, limb_weights=limb_weights,
            add_probs=True, outfile="cole_mult_hmm_results.csv"
        )

    elif args.algorithm == 'CWH':
        # weighted-consensus Cole–Kripke + HMM
        try:
            from apply_cole_kripke import apply_cole_kripke_mult_weighted
        except ImportError:
            sys.exit("Error: Could not import 'apply_cole_kripke_mult_weighted'.")
        try:
            from hmm_postprocess import apply_hmm_over_cole
        except ImportError:
            sys.exit("Error: Could not import 'apply_hmm_over_cole' (hmm_postprocess.py missing?).")

        ck = apply_cole_kripke_mult_weighted(df, args.limbs)
        limb_weights = parse_weights(args.weights, args.limbs)
        result = apply_hmm_over_cole(
            ck, num_limbs=args.limbs, tau=args.tau, alpha=args.alpha,
            p_sw=args.p_sw, p_ws=args.p_ws, limb_weights=limb_weights,
            add_probs=True, outfile="cole_weighted_mult_hmm_results.csv"
        )
    # --------------------------------------------------------------------

    elif args.algorithm == 'S':
        try:
            from traditional_methods import apply_sadeh_combined
        except ImportError:
            sys.exit("Error: Could not import 'apply_sadeh_combined'.")
        result = apply_sadeh_combined(df)

    elif args.algorithm == 'TRO':
        try:
            from traditional_methods import apply_troiano_combined
        except ImportError:
            sys.exit("Error: Could not import 'apply_troiano_combined'.")
        result = apply_troiano_combined(df)

    elif args.algorithm == 'CHO':
        try:
            from traditional_methods import apply_choi_combined
        except ImportError:
            sys.exit("Error: Could not import 'apply_choi_combined'.")
        result = apply_choi_combined(df)

    else:
        sys.exit(
            f"Error: Unknown algorithm '{args.algorithm}'. "
            "Supported values are: C, CM, CMM, CW, S, TRO, CHO."
        )

    print("Algorithm Output:")
    print(result)

if __name__ == "__main__":
    main()