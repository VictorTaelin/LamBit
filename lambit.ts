// LamBit.ts
// =========
// A minimal functional language based on binary pattern-matching case-trees.
//
//     Func ::=
//       | Get ::= "!" Func
//       | Mat ::= "λ" "{" "0" ":" Func ";" "1" ":" Func ";" "}"
//       | Lam ::= "λ" Name "." Func
//       | Use ::= "λ" "()" "." Func
//       | Ret ::= Term
//
//     Term ::=
//       | Var ::= Name
//       | Bt0 ::= "0"
//       | Bt1 ::= "1"
//       | Nul ::= "()"
//       | Tup ::= "(" Term "," Term ")"
//       | Rec ::= "~" Term
//
// LamBit programs are composed by a single "top-level" function. Computation
// happens by recursively pattern-matching that function against a single input.
// 
// LamBit is Turing-complete. For example, consider the following Haskell program:
// 
//     data Nat    = Z | S Nat
//     add Z     b = b
//     add (S a) b = S (add a b)
//     mul Z     b = Z
//     mul (S a) b = add b (mul a b)
// 
// It can be expressed on LamBit as:
//
//     F = ! λ{
//       0: ! λ{ // computes addition
//         0: λ(). b
//         1: λa. λb. (1,~(0,(a,b)))
//       }
//       1: ! λ{ // computes multiplication
//         0: λb. 0{}
//         1: λa. λb. ~(0,(b,~(1,(a,b))))
//       }
//     }
//
// Where natural numbers are encoded as:
//
//     Nat ::=
//     | zero ::= (0,())
//     | succ ::= (1,Nat)
//     (aliasing 3n as (1,(1,(1,(0,())))), and so on)
//
// For instance, ~(1,(2n,3n)) evaluates 2n*3n to 6n:
//
//     ~(1,(2n,3n))
//     = ~(0,(3n, ~(1,(1n,3n))))
//     = ~(0,(3n, ~(0,(3n, ~(1,((0,()),3n))))))
//     = ~(0,(3n, ~(0,(3n, (0,())))))
//     = ~(0,(3n, 3n))
//     = (1,~(0,(2n,3n)))
//     = (1,(1,~(0,(1n,3n))))
//     = (1,(1,(1,~(0,((0,()),3n)))))
//     = (1,(1,(1,3n)))
//     = 6n

// Types
// -----

// Function-side: patterns and lambdas
type Get  = { $: "Get", fun: Func };
type Mat  = { $: "Mat", zro: Func, one: Func };
type Lam  = { $: "Lam", nam: string, bod: (x: Term) => Func };
type Use  = { $: "Use", bod: Func };
type Ret  = { $: "Ret", trm: Term };
type Func = Get | Mat | Lam | Use | Ret;

// Data-side: values and expressions
type Var  = { $: "Var", nam: string };
type Bt0  = { $: "Bt0" };
type Bt1  = { $: "Bt1" };
type Nul  = { $: "Nul" };
type Tup  = { $: "Tup", fst: Term, snd: Term };
type Rec  = { $: "Rec", arg: Term };
type Term = Var | Bt0 | Bt1 | Nul | Tup | Rec;

// Interaction stats
type Stats = { app_fun: number, app_lam: number, app_mat: number, app_get: number, app_use: number };

// Constructors
const Get = (fun)      => ({ $: "Get", fun }) as Func;
const Mat = (zro, one) => ({ $: "Mat", zro, one }) as Func;
const Lam = (nam, bod) => ({ $: "Lam", nam, bod }) as Func;
const Use = (bod)      => ({ $: "Use", bod }) as Func;
const Ret = (trm)      => ({ $: "Ret", trm }) as Func;
const Var = (nam)      => ({ $: "Var", nam }) as Term;
const Bt0 = ()         => ({ $: "Bt0" }) as Term;
const Bt1 = ()         => ({ $: "Bt1" }) as Term;
const Nul = ()         => ({ $: "Nul" }) as Term;
const Tup = (fst, snd) => ({ $: "Tup", fst, snd }) as Term;
const Rec = (arg)      => ({ $: "Rec", arg }) as Term;

// Creates a fresh stats counter
function new_stats(): Stats {
  return { app_fun: 0, app_lam: 0, app_mat: 0, app_get: 0, app_use: 0 };
}

// Parser
// ------

type Parser = { src: string, idx: number };

// Skips whitespace
function skip(p: Parser): void {
  while (p.idx < p.src.length && /\s/.test(p.src[p.idx])) {
    p.idx++;
  }
}

// Tries to consume a string, returns success
function match(p: Parser, str: string): boolean {
  skip(p);
  if (p.src.startsWith(str, p.idx)) {
    p.idx += str.length;
    return true;
  }
  return false;
}

// Consumes a string or throws
function expect(p: Parser, str: string): void {
  if (!match(p, str)) {
    throw new Error(`Expected '${str}' at index ${p.idx}`);
  }
}

// Peeks at current char after skipping whitespace
function peek(p: Parser): string {
  skip(p);
  return p.src[p.idx] || "";
}

// Parses an alphanumeric name
function parse_name(p: Parser): string {
  skip(p);
  var start = p.idx;
  while (p.idx < p.src.length && /[a-zA-Z_0-9]/.test(p.src[p.idx])) {
    p.idx++;
  }
  if (p.idx === start) {
    throw new Error(`Expected name at index ${p.idx}`);
  }
  return p.src.slice(start, p.idx);
}

// Checks if current position is "()" (null literal)
function is_nul(p: Parser): boolean {
  skip(p);
  return p.src[p.idx] === "(" && p.src[p.idx + 1] === ")";
}

// Checks if current position starts a tuple: "(" but not "()"
function is_tup(p: Parser): boolean {
  skip(p);
  return p.src[p.idx] === "(" && p.src[p.idx + 1] !== ")";
}

// Parses a Func with a name context for HOAS
function parse_func(p: Parser, ctx: Map<string, Term>): Func {
  skip(p);
  // Get: ! Func
  if (match(p, "!")) {
    var fun = parse_func(p, ctx);
    return Get(fun);
  }
  // Mat or Lam or Use
  if (match(p, "λ")) {
    // Mat: λ{ 0: ... 1: ... }
    if (match(p, "{")) {
      expect(p, "0");
      expect(p, ":");
      var zro = parse_func(p, ctx);
      match(p, ";");
      expect(p, "1");
      expect(p, ":");
      var one = parse_func(p, ctx);
      match(p, ";");
      expect(p, "}");
      return Mat(zro, one);
    }
    // Use: λ(). Func
    if (match(p, "()")) {
      expect(p, ".");
      var bod = parse_func(p, ctx);
      return Use(bod);
    }
    // Lam: λName. Func
    var nam = parse_name(p);
    expect(p, ".");
    return parse_func_lam(p, ctx, nam);
  }
  // Ret: Term
  var trm = parse_term(p, ctx);
  return Ret(trm);
}

// Parses a lambda, advancing the parser correctly
function parse_func_lam(p: Parser, ctx: Map<string, Term>, nam: string): Func {
  // Parse body once with sentinel to find end position
  var sentinel   = Var("$" + nam);
  var ctx2       = new Map(ctx);
  ctx2.set(nam, sentinel);
  var body_start = p.idx;
  parse_func(p, ctx2);
  var body_end   = p.idx;
  // Return a Lam that re-parses body with the actual argument
  return Lam(nam, (x: Term) => {
    var ctx3 = new Map(ctx);
    ctx3.set(nam, x);
    var bp = { src: p.src, idx: body_start };
    return parse_func(bp, ctx3);
  });
}

// Parses a Term with a name context
function parse_term(p: Parser, ctx: Map<string, Term>): Term {
  skip(p);
  // Rec: ~Term
  if (match(p, "~")) {
    var arg = parse_term(p, ctx);
    return Rec(arg);
  }
  // Nul: ()
  if (is_nul(p)) {
    match(p, "()");
    return Nul();
  }
  // Tup: (Term, Term)
  if (is_tup(p)) {
    match(p, "(");
    var fst = parse_term(p, ctx);
    expect(p, ",");
    var snd = parse_term(p, ctx);
    expect(p, ")");
    return Tup(fst, snd);
  }
  // Bt0: 0
  if (peek(p) === "0" && !/[a-zA-Z_0-9{]/.test(p.src[p.idx + 1] || "")) {
    p.idx++;
    return Bt0();
  }
  // Bt1: 1
  if (peek(p) === "1" && !/[a-zA-Z_0-9{]/.test(p.src[p.idx + 1] || "")) {
    p.idx++;
    return Bt1();
  }
  // Var: Name
  var nam = parse_name(p);
  var val = ctx.get(nam);
  if (val === undefined) {
    throw new Error(`Unbound variable '${nam}'`);
  }
  return val;
}

// Evaluation
// ----------

// Feeds a single term into a Func (handles Get, Mat, Lam, Use)
function feed(prog: Func, func: Func, term: Term, stats: Stats): Func {
  switch (func.$) {
    case "Get": {
      if (term.$ !== "Tup") {
        throw new Error("Get expected a Tup");
      }
      stats.app_get++;
      var result = feed(prog, func.fun, term.fst, stats);
      result = feed(prog, result, term.snd, stats);
      return result;
    }
    case "Lam": {
      stats.app_lam++;
      return func.bod(term);
    }
    case "Mat": {
      if (term.$ !== "Bt0" && term.$ !== "Bt1") {
        throw new Error("Mat expected a Bt0 or Bt1");
      }
      stats.app_mat++;
      return term.$ === "Bt0" ? func.zro : func.one;
    }
    case "Use": {
      if (term.$ !== "Nul") {
        throw new Error("Use expected a Nul");
      }
      stats.app_use++;
      return func.bod;
    }
    default: {
      throw new Error("Cannot feed into a Ret");
    }
  }
}

// Reduces a term to normal form
function eval_term(prog: Func, term: Term, stats: Stats): Term {
  switch (term.$) {
    case "Var": {
      return term;
    }
    case "Bt0": {
      return term;
    }
    case "Bt1": {
      return term;
    }
    case "Nul": {
      return term;
    }
    case "Tup": {
      var fst = eval_term(prog, term.fst, stats);
      var snd = eval_term(prog, term.snd, stats);
      return Tup(fst, snd);
    }
    case "Rec": {
      stats.app_fun++;
      var arg  = eval_term(prog, term.arg, stats);
      var func = feed(prog, prog, arg, stats);
      if (func.$ !== "Ret") {
        throw new Error("Expected Ret after full application");
      }
      return eval_term(prog, func.trm, stats);
    }
  }
}

// Show
// ----

// Converts a term to a string
function show_term(term: Term): string {
  switch (term.$) {
    case "Var": {
      return term.nam;
    }
    case "Bt0": {
      return "0";
    }
    case "Bt1": {
      return "1";
    }
    case "Nul": {
      return "()";
    }
    case "Tup": {
      var fst = show_term(term.fst);
      var snd = show_term(term.snd);
      return `(${fst},${snd})`;
    }
    case "Rec": {
      var arg = show_term(term.arg);
      return `~${arg}`;
    }
  }
}

// Converts a func to a string
function show_func(func: Func, dep: number = 0): string {
  switch (func.$) {
    case "Get": {
      var fun = show_func(func.fun, dep);
      return `! ${fun}`;
    }
    case "Mat": {
      var zro = show_func(func.zro, dep);
      var one = show_func(func.one, dep);
      return `λ{ 0: ${zro}; 1: ${one} }`;
    }
    case "Lam": {
      var nam = func.nam || String.fromCharCode(97 + dep);
      var bod = show_func(func.bod(Var(nam)), dep + 1);
      return `λ${nam}. ${bod}`;
    }
    case "Use": {
      var bod = show_func(func.bod, dep);
      return `λ(). ${bod}`;
    }
    case "Ret": {
      return show_term(func.trm);
    }
  }
}

// Formats stats as a report string
function show_stats(stats: Stats): string {
  var total = stats.app_fun + stats.app_lam + stats.app_mat + stats.app_get + stats.app_use;
  var lines = [
    `Interactions: ${total}`,
    `- APP-FUN: ${stats.app_fun}`,
    `- APP-LAM: ${stats.app_lam}`,
    `- APP-MAT: ${stats.app_mat}`,
    `- APP-GET: ${stats.app_get}`,
    `- APP-USE: ${stats.app_use}`,
  ];
  return lines.join("\n");
}

// Test
// ----

// Decrements 65535 til it hits 0.
// 
// Bits ::=
// | []     ::= (0,())
// | 0 : xs ::= (1,(0,xs))
// | 1 : xs ::= (1,(1,xs))
// 
// Bool ::=
// | False ::= 0
// | True  ::= 1
//
// main x        = cond (is0 x) x
// is0 []        = True
// is0 (0 : xs)  = is0 xs
// is0 (1 : xs)  = False
// dec []        = []
// dec (0 : xs)  = 1 : dec xs
// dec (1 : xs)  = 0 : xs
// cond False x  = main (dec x)
// cond True  x  = []

function test() {
  // F = ! λ{
  //   0: ! λ{
  //     0: λx. ~(1,(1,(~(0,(1,x)),x)))       // main x = cond (is0 x) x
  //     1: ! λ{                               // is0:
  //       0: λ(). 1                            //   is0 [] = True
  //       1: ! λ{                              //   is0 (b:xs):
  //         0: λxs. ~(0,(1,xs))                //     is0 (0:xs) = is0 xs
  //         1: λxs. 0                          //     is0 (1:xs) = False
  //       }
  //     }
  //   }
  //   1: ! λ{
  //     0: ! λ{                               // dec:
  //       0: λ(). (0,())                       //   dec [] = []
  //       1: ! λ{                              //   dec (b:xs):
  //         0: λxs. (1,(1,~(1,(0,xs))))        //     dec (0:xs) = 1 : dec xs
  //         1: λxs. (1,(0,xs))                 //     dec (1:xs) = 0 : xs
  //       }
  //     }
  //     1: ! λ{                               // cond:
  //       0: λx. ~(0,(0,~(1,(0,x))))           //   cond False x = main (dec x)
  //       1: λx. (0,())                        //   cond True  x = []
  //     }
  //   }
  // }
  var prog_src = `! λ{
    0: ! λ{
      0: λx. ~(1,(1,(~(0,(1,x)),x)))
      1: ! λ{
        0: λ(). 1
        1: ! λ{
          0: λxs. ~(0,(1,xs))
          1: λxs. 0
        }
      }
    }
    1: ! λ{
      0: ! λ{
        0: λ(). (0,())
        1: ! λ{
          0: λxs. (1,(1,~(1,(0,xs))))
          1: λxs. (1,(0,xs))
        }
      }
      1: ! λ{
        0: λx. ~(0,(0,~(1,(0,x))))
        1: λx. (0,())
      }
    }
  }`;

  // Parse the program
  var pp   = { src: prog_src, idx: 0 };
  var prog = parse_func(pp, new Map());
  console.log("prog:", show_func(prog));

  // Build 65535 as 16 one-bits (LSB-first): (1,(1,(1,(1,...(1,(1,(0,())))...))))
  var n = "(0,())";
  for (var i = 0; i < 20; i++) {
    n = "(1,(1," + n + "))";
  }
  // Call: ~(0,(0,<65535>)) = main(65535)
  var input_src = "~(0,(0," + n + "))";
  var input     = parse_term({src: input_src, idx: 0}, new Map());

  // Evaluate and print
  var stats  = new_stats();
  var result = eval_term(prog, input, stats);
  console.log(show_term(result));
  // Expected: (0,()) (halted after counting down to zero)

  // Print interaction stats
  console.log(show_stats(stats));
}

test();

// That works very well - good job!
// Sadly, the TS interpreter is too slow. The term above returns:
// (0,())
// Interactions: 56622872
// - APP-FUN: 6291433
// - APP-LAM: 6291432
// - APP-MAT: 22020003
// - APP-GET: 22020003
// - APP-USE: 1
// In 5 seconds. That means it achieves only about ~11m interactions/s. 

// Now, your goal is to extend LamBit with a fast, memory-efficient C compiler.
// To achieve that, we will represent memory using U16 Ptrs, where:
// Ptr ::= Ctr | Loc
// Ctr ::= NUL | TUP | BT0 | BT1
// Loc ::= 14-bit address
// The top-level program will be *fully compiled* to a single, efficient, native
// C function. That means we don't need to implement VAR or REC as Term
// variants, since these are just part of the compiled C function, and not valid
// as runtime terms. We also need to implement a very fast allocator and
// ref-counted garbage collector.  The allocator will be dead simple: it will
// just be a bump allocator that wraps around the complete heap (which is a
// buffer with 2^14 u16 Terms) seeking an empty slot (==0). The garbage
// collector will simply free() a Term and its children recursively when its
// ref-count goes to 0. The ref-count of a reference decreases when it is passed
// to a lambda that doesn't use it on the returned Term.
// The C compiler will receive a LamBit program as its input, and output a C
// program that applies it to an input, and prints the result, including stats,
// as fast as possible. Example usage:
// $ lambit main.lb -o main
// $ ./main "~(0,(0,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(1,(0,())))))))))))))))))))))))))))))))))))))))))))"
// should output exactly the same as test() above, but in *much* less time.
// Your ultimate goal is to:
// - implement the compiler correctly
// - make it as fast as you can
// Start working on this now.
