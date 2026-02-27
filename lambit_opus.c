//./lambit.ts//

//// That works very well - good job!
//// Sadly, the TS interpreter is too slow. The term above returns:
////prog: λ! λ{ 0: λ! λ{ 0: λx. ~(0,(1,(~(1,(1,x)),(0,())))); 1: λ! λ! λ{ 0: λ(). λzs. zs; 1: λ! λ{ 0: λxs. λzs. ~(0,(1,(xs,(1,(0,zs))))); 1: λxs. λzs. ~(0,(0,~(1,(0,(zs,(1,(1,xs))))))) } } }; 1: λ! λ{ 0: λ! λ! λ{ 0: λ(). λys. ys; 1: λ! λx. λxs. λys. (1,(x,~(1,(0,(xs,ys))))) }; 1: λ! λ{ 0: λ(). (0,()); 1: λ! λ{ 0: λxs. (1,(1,~(1,(1,xs)))); 1: λxs. (1,(0,xs)) } } } }
////(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(0,())))))))))))))))))))))))))))))))))))))))))
////Interactions: 75496948
////- APP-FUN: 7339984
////- APP-LAM: 11534243
////- APP-MAT: 25165656
////- APP-GET: 30408490
////- APP-USE: 1048575
//// In ~8 seconds. That means it achieves only about ~9m interactions/s. 
//
//// Now, your goal is to port LamBit to a fast, memory-efficient C runtime.
//// To achieve that, we will represent Terms in memory via U16 Ptrs, where:
//// - Ptr ::= Ctr & Loc
//// - Ctr ::= NUL | TUP | BT0 | BT1
//// - Loc ::= 14-bit address
//// We also include a super fast bump allocator, which pre-allocs a buffer with
//// 2^14 u16's to store memory terms (using hints to pin it to the L1 cache), and
//// works by incrementing the allocation cursor K, and checking if heap[K] is 0.
//// We also include a garbage collector, which is triggered whenever a lambda is
//// applied to a term, yet the lambda doesn't use it on the returned expression.
//// This will zero that term and all its descendants from memory. Note that this
//// is only valid if the input program is affine. We assume that's the case.
//// Note that, in C, in-memory terms won't include VAR and RET; these belong to
//// the Funcs, which are represented via a compact buffer, with 1 byte per node:
//// - 0x00      ::= Nul
//// - 0x01      ::= Bt0
//// - 0x02      ::= Bt1
//// - 0x03      ::= Lam(Func)
//// - 0x04      ::= Use(Func)
//// - 0x05      ::= Get(Func)
//// - 0x06      ::= Ret(Func)
//// - 0x08~0x1F ::= Var(BIdx)
//// - 0x20~0x8F ::= Mat(Func,Func)
//// - 0x90~0xFF ::= Tup(Func,Func)
//// Here, variables are Bruijn levels (up to 23), and Mat/Tup are represented by
//// a range of potential bytes. We use that to embbed the delta position towards
//// the second branch/field in the buffer - otherwise, we would not know where to
//// locate it.
//// The C runtime parses the input source into the compact buffer, and the
//// evaluator reads only from it. The normalizer uses a simple substitution
//// buffer to store the local context. Our goal is for it to be as fast as
//// possible. This will be running in a cluster, so, each flop counts.
//// Implement the fast C runtime below:
//
//// LamBit — Fast C Runtime
//// Compile: gcc -O3 -march=native -o lambit lambit.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

// ---- Term Representation (u16) ----
// Tag (2 bits) | Loc (14 bits)
#define TAG_NUL 0
#define TAG_BT0 1
#define TAG_BT1 2
#define TAG_TUP 3

#define MK(tag, loc)  ((uint16_t)(((tag) << 14) | ((loc) & 0x3FFF)))
#define TAG(p)        ((p) >> 14)
#define LOC(p)        ((p) & 0x3FFF)

#define PTR_NUL MK(TAG_NUL, 1)
#define PTR_BT0 MK(TAG_BT0, 0)
#define PTR_BT1 MK(TAG_BT1, 0)

// ---- Heap (free-list allocator) ----
#define HEAP_SIZE (1 << 14)
static uint16_t heap[HEAP_SIZE] __attribute__((aligned(64)));
static uint32_t alloc_cursor = 1;
static uint16_t free_head = 0;

static inline uint16_t heap_fst(uint16_t p) { return heap[LOC(p)]; }
static inline uint16_t heap_snd(uint16_t p) { return heap[LOC(p) + 1]; }

// Allocates a 2-cell pair from the free list, or bumps
static inline uint32_t alloc2(void) {
  if (free_head != 0) {
    uint32_t loc = free_head;
    free_head = heap[loc];
    return loc;
  }
  uint32_t loc = alloc_cursor;
  alloc_cursor = loc + 2;
  return loc;
}

// Returns a 2-cell pair to the free list
static inline void free_pair(uint32_t loc) {
  heap[loc] = free_head;
  free_head = (uint16_t)loc;
}

// Allocates a tuple on the heap
static inline uint16_t make_tup(uint16_t fst, uint16_t snd) {
  uint32_t loc = alloc2();
  heap[loc]     = fst;
  heap[loc + 1] = snd;
  return MK(TAG_TUP, loc);
}

// Frees an unreferenced term tree
static void free_term(uint16_t p) {
  while (TAG(p) == TAG_TUP) {
    uint32_t loc = LOC(p);
    uint16_t f = heap[loc];
    uint16_t s = heap[loc + 1];
    free_pair(loc);
    if (TAG(f) == TAG_TUP) free_term(f);
    p = s;
  }
}

// ---- Bytecode ----
#define OP_NUL 0x00
#define OP_BT0 0x01
#define OP_BT1 0x02
#define OP_LAM 0x03
#define OP_USE 0x04
#define OP_GET 0x05
#define OP_REC 0x06
#define OP_ERA 0x07
#define OP_VAR_BASE 0x08
#define OP_VAR_MAX  0x1F
#define OP_MAT_BASE 0x20
#define OP_MAT_MAX  0x8F
#define OP_TUP_BASE 0x90
#define OP_TUP_MAX  0xFF

#define MAX_CODE 8192
static uint8_t code[MAX_CODE];
static uint32_t code_len = 0;


// ---- Stats ----
static uint64_t stat_fun = 0;
static uint64_t stat_lam = 0;
static uint64_t stat_mat = 0;
static uint64_t stat_get = 0;
static uint64_t stat_use = 0;

// ---- Substitution stack (de Bruijn indices) ----
// Var(0) = most recently bound = sub_stack[sub_sp - 1]
// Var(k) = sub_stack[sub_sp - 1 - k]
// A slot of 0 means "consumed" (PTR_NUL is non-zero).
#define MAX_SUB (1 << 12)
static uint16_t sub_stack[MAX_SUB];
static uint32_t sub_sp = 0;
static uint32_t max_sub_sp = 0;

// ---- Continuation stack (packed u32) ----
// Layout: tag(16) | val(16)
// val is snd_pc, fst_val, or saved_sp (all fit in 16 bits).
#define CONT_TUP_SND  1
#define CONT_TUP_DONE 2
#define CONT_REC      3
#define CONT_REC_GC   4
#define CONT_TAIL_REC 5

#define CONT_MAKE(tag, val) (((uint32_t)(tag) << 16) | ((uint32_t)(val) & 0xFFFF))
#define CONT_TAG_OF(c)      ((c) >> 16)
#define CONT_VAL_OF(c)      ((uint16_t)((c) & 0xFFFF))

#define MAX_CONT (1 << 12)
static uint32_t cont_stack[MAX_CONT];
static uint32_t cont_sp = 0;
static uint32_t max_cont_sp = 0;

// ---- Bytecode Compiler (source -> bytecode) ----

typedef struct { const char *src; int idx; int len; } Src;

static void src_skip(Src *s) {
  while (s->idx < s->len && (s->src[s->idx]==' ' || s->src[s->idx]=='\n' ||
         s->src[s->idx]=='\r' || s->src[s->idx]=='\t'))
    s->idx++;
}

static int src_match(Src *s, const char *str) {
  src_skip(s);
  int n = (int)strlen(str);
  if (s->idx + n <= s->len && memcmp(s->src + s->idx, str, n) == 0) {
    s->idx += n;
    return 1;
  }
  return 0;
}

static void src_expect(Src *s, const char *str) {
  if (!src_match(s, str)) {
    fprintf(stderr, "Expected '%s' at index %d\n", str, s->idx);
    exit(1);
  }
}

static char src_peek(Src *s) {
  src_skip(s);
  return s->idx < s->len ? s->src[s->idx] : '\0';
}

static int is_name_char(char c) {
  return (c>='a'&&c<='z') || (c>='A'&&c<='Z') || (c>='0'&&c<='9') || c=='_';
}

#define MAX_NAMES 64
static char  cname_buf[MAX_NAMES][32];
static int   cname_used[MAX_NAMES];
static int   cname_depth = 0;

static void parse_name_into(Src *s, char *out, int maxlen) {
  src_skip(s);
  int start = s->idx;
  while (s->idx < s->len && is_name_char(s->src[s->idx])) s->idx++;
  int n = s->idx - start;
  if (n == 0) { fprintf(stderr, "Expected name at %d\n", s->idx); exit(1); }
  if (n >= maxlen) n = maxlen - 1;
  memcpy(out, s->src + start, n);
  out[n] = '\0';
}

// Find name, return de Bruijn INDEX (0 = innermost)
static int find_name_dbi(const char *nm) {
  for (int i = cname_depth - 1; i >= 0; i--) {
    if (strcmp(cname_buf[i], nm) == 0)
      return (cname_depth - 1) - i;
  }
  return -1;
}

static void emit(uint8_t b) {
  if (code_len >= MAX_CODE) { fprintf(stderr, "Code overflow\n"); exit(1); }
  code[code_len++] = b;
}

static void compile_func(Src *s);
static void compile_term(Src *s);

static int src_is_nul(Src *s) {
  src_skip(s);
  return s->idx+1 < s->len && s->src[s->idx]=='(' && s->src[s->idx+1]==')';
}

static int src_is_tup(Src *s) {
  src_skip(s);
  return s->idx < s->len && s->src[s->idx]=='(' && (s->idx+1 >= s->len || s->src[s->idx+1]!=')');
}

// Match UTF-8 λ followed by !
static int match_lambda_bang(Src *s) {
  src_skip(s);
  if (s->idx + 2 < s->len &&
      (uint8_t)s->src[s->idx] == 0xCE &&
      (uint8_t)s->src[s->idx+1] == 0xBB &&
      s->src[s->idx+2] == '!') {
    s->idx += 3;
    return 1;
  }
  return 0;
}

// Match UTF-8 λ
static int match_lambda(Src *s) {
  src_skip(s);
  if (s->idx + 1 < s->len &&
      (uint8_t)s->src[s->idx] == 0xCE &&
      (uint8_t)s->src[s->idx+1] == 0xBB) {
    s->idx += 2;
    return 1;
  }
  return 0;
}

static void compile_func(Src *s) {
  src_skip(s);

  // Get: λ! Func
  if (match_lambda_bang(s)) {
    emit(OP_GET);
    compile_func(s);
    return;
  }

  // Lambda/Mat/Use
  if (match_lambda(s)) {
    // Mat: λ{ 0: ... ; 1: ... }
    if (src_match(s, "{")) {
      src_expect(s, "0");
      src_expect(s, ":");
      uint32_t mat_pos = code_len;
      emit(0); // placeholder
      uint32_t zero_start = code_len;
      compile_func(s);
      uint32_t zero_end = code_len;
      src_match(s, ";");
      src_expect(s, "1");
      src_expect(s, ":");
      compile_func(s);
      src_match(s, ";");
      src_expect(s, "}");
      uint32_t delta = zero_end - zero_start;
      if (delta > (uint32_t)(OP_MAT_MAX - OP_MAT_BASE)) {
        fprintf(stderr, "Mat delta too large: %u\n", delta);
        exit(1);
      }
      code[mat_pos] = (uint8_t)(OP_MAT_BASE + delta);
      return;
    }
    // Use: λ(). Func
    if (src_match(s, "()")) {
      src_expect(s, ".");
      emit(OP_USE);
      compile_func(s);
      return;
    }
    // Lam: λName. Func (or ERA if var unused)
    char nm[32];
    parse_name_into(s, nm, 32);
    src_expect(s, ".");
    uint32_t lam_pos = code_len;
    emit(0); // placeholder: patched to LAM or ERA
    strcpy(cname_buf[cname_depth], nm);
    cname_used[cname_depth] = 0;
    cname_depth++;
    compile_func(s);
    cname_depth--;
    code[lam_pos] = cname_used[cname_depth] ? OP_LAM : OP_ERA;
    return;
  }

  // Otherwise: a term expression (Ret is implicit)
  compile_term(s);
}

static void compile_term(Src *s) {
  src_skip(s);

  // Rec: ~Term
  if (src_match(s, "~")) {
    emit(OP_REC);
    compile_term(s);
    return;
  }

  // Nul: ()
  if (src_is_nul(s)) {
    src_match(s, "()");
    emit(OP_NUL);
    return;
  }

  // Tup: (Term, Term)
  if (src_is_tup(s)) {
    src_match(s, "(");
    uint32_t tup_pos = code_len;
    emit(0); // placeholder
    uint32_t fst_start = code_len;
    compile_term(s);
    uint32_t fst_end = code_len;
    src_expect(s, ",");
    compile_term(s);
    src_expect(s, ")");
    uint32_t delta = fst_end - fst_start;
    if (delta > (uint32_t)(OP_TUP_MAX - OP_TUP_BASE)) {
      fprintf(stderr, "Tup delta too large: %u\n", delta);
      exit(1);
    }
    code[tup_pos] = (uint8_t)(OP_TUP_BASE + delta);
    return;
  }

  // Bt0: 0 (not followed by alnum)
  char ch = src_peek(s);
  if (ch == '0' && (s->idx + 1 >= s->len || !is_name_char(s->src[s->idx + 1]))) {
    s->idx++;
    emit(OP_BT0);
    return;
  }

  // Bt1: 1 (not followed by alnum)
  if (ch == '1' && (s->idx + 1 >= s->len || !is_name_char(s->src[s->idx + 1]))) {
    s->idx++;
    emit(OP_BT1);
    return;
  }

  // Var: Name -> de Bruijn index
  char nm[32];
  parse_name_into(s, nm, 32);
  int dbi = find_name_dbi(nm);
  if (dbi < 0) { fprintf(stderr, "Unbound variable '%s'\n", nm); exit(1); }
  cname_used[cname_depth - 1 - dbi] = 1;
  if (dbi > (OP_VAR_MAX - OP_VAR_BASE)) {
    fprintf(stderr, "Variable index too large: %d\n", dbi);
    exit(1);
  }
  emit((uint8_t)(OP_VAR_BASE + dbi));
}

// ---- Disassembler ----
static void print_code(uint32_t pc, int indent) {
  uint8_t op = code[pc];
  if (op == OP_NUL) { printf("NUL"); return; }
  if (op == OP_BT0) { printf("BT0"); return; }
  if (op == OP_BT1) { printf("BT1"); return; }
  if (op >= OP_VAR_BASE && op <= OP_VAR_MAX) { printf("VAR(%d)", op - OP_VAR_BASE); return; }
  if (op == OP_LAM) { printf("LAM "); print_code(pc+1, indent); return; }
  if (op == OP_ERA) { printf("ERA "); print_code(pc+1, indent); return; }
  if (op == OP_USE) { printf("USE "); print_code(pc+1, indent); return; }
  if (op == OP_GET) { printf("GET "); print_code(pc+1, indent); return; }
  if (op == OP_REC) { printf("REC "); print_code(pc+1, indent); return; }
  if (op >= OP_MAT_BASE && op <= OP_MAT_MAX) {
    uint32_t d = op - OP_MAT_BASE;
    printf("MAT(d=%u) {0: ", d);
    print_code(pc+1, indent+2);
    printf("; 1: ");
    print_code(pc+1+d, indent+2);
    printf("}");
    return;
  }
  if (op >= OP_TUP_BASE && op <= OP_TUP_MAX) {
    uint32_t d = op - OP_TUP_BASE;
    printf("TUP(d=%u) (", d);
    print_code(pc+1, indent+2);
    printf(", ");
    print_code(pc+1+d, indent+2);
    printf(")");
    return;
  }
  printf("?0x%02x", op);
}

// ---- Feed: feeds a heap term into the func at pc ----
// Pushes LAM bindings onto sub_stack. Returns pc of resulting expression.
// Iterative on snd (via GET), recursive only on fst.
static uint32_t feed_term(uint32_t pc, uint16_t term) {
  for (;;) {
    uint8_t op = code[pc];

    if (op == OP_GET) {
      stat_get++;
      uint32_t loc = LOC(term);
      uint16_t fst = heap[loc];
      uint16_t snd = heap[loc + 1];
      free_pair(loc);
      // Fast path: GET followed by MAT (most common pattern)
      uint8_t nxt = code[pc + 1];
      if (nxt >= OP_MAT_BASE && nxt <= OP_MAT_MAX) {
        stat_mat++;
        uint32_t delta = nxt - OP_MAT_BASE;
        pc = (TAG(fst) == TAG_BT0) ? pc + 2 : pc + 2 + delta;
        term = snd;
        continue;
      }
      // Slow path: recurse on fst, iterate on snd
      pc = feed_term(pc + 1, fst);
      term = snd;
      continue;
    }

    if (op == OP_LAM) {
      stat_lam++;
      sub_stack[sub_sp++] = term;
      return pc + 1;
    }

    if (op == OP_ERA) {
      stat_lam++;
      free_term(term);
      sub_stack[sub_sp++] = 0; // placeholder for index alignment
      return pc + 1;
    }

    if (op >= OP_MAT_BASE && op <= OP_MAT_MAX) {
      stat_mat++;
      uint32_t delta = op - OP_MAT_BASE;
      return (TAG(term) == TAG_BT0) ? pc + 1 : pc + 1 + delta;
    }

    if (op == OP_USE) {
      stat_use++;
      return pc + 1;
    }

    fprintf(stderr, "Cannot feed into opcode 0x%02x at pc=%u\n", op, pc);
    exit(1);
  }
}

// ---- Evaluator ----
// Goto dispatch, packed u32 conts, TUP(d=1) fusion, deforestation.

static uint16_t __attribute__((hot)) eval_iterative(uint32_t start_pc) {
  uint32_t pc  = start_pc;
  uint16_t val = 0;
  uint32_t saved_sp;
  const uint8_t *cc = code;

  // ---- EVAL: dispatch on opcode ----
  eval: {
    uint8_t op = cc[pc];

    // Most common: TUP (opcode >= 0x90)
    if (__builtin_expect(op >= OP_TUP_BASE, 1)) {
      uint32_t delta = op - OP_TUP_BASE;
      if (__builtin_expect(delta == 1, 1)) {
        uint8_t fst_op = cc[pc + 1];
        uint16_t fst_val;
        if (fst_op == OP_BT0)      { fst_val = PTR_BT0; }
        else if (fst_op == OP_BT1) { fst_val = PTR_BT1; }
        else if (fst_op == OP_NUL) { fst_val = PTR_NUL; }
        else if (fst_op >= OP_VAR_BASE && fst_op <= OP_VAR_MAX) {
          uint32_t idx = sub_sp - 1 - (fst_op - OP_VAR_BASE);
          fst_val = sub_stack[idx];
          sub_stack[idx] = 0;
        } else {
          goto tup_slow;
        }
        cont_stack[cont_sp++] = CONT_MAKE(CONT_TUP_DONE, fst_val);
        pc += 2;
        goto eval;
      }
      tup_slow:
      cont_stack[cont_sp++] = CONT_MAKE(CONT_TUP_SND, pc + 1 + delta);
      pc += 1;
      goto eval;
    }

    if (op == OP_REC) {
      stat_fun++;
      cont_stack[cont_sp++] = CONT_MAKE(CONT_REC, sub_sp);
      pc += 1;
      goto eval;
    }
    if (op == OP_NUL) { val = PTR_NUL; goto ret; }
    if (op == OP_BT0) { val = PTR_BT0; goto ret; }
    if (op == OP_BT1) { val = PTR_BT1; goto ret; }
    if (op >= OP_VAR_BASE) {
      uint32_t idx = sub_sp - 1 - (op - OP_VAR_BASE);
      val = sub_stack[idx];
      sub_stack[idx] = 0;
      goto ret;
    }
    __builtin_unreachable();
  }

  // ---- RETURN: process continuation ----
  ret: {
    if (__builtin_expect(cont_sp == 0, 0)) return val;
    uint32_t c = cont_stack[cont_sp - 1];

    // Most common: TUP_DONE chain
    if (__builtin_expect(CONT_TAG_OF(c) == CONT_TUP_DONE, 1)) {
      for (;;) {
        cont_sp--;
        if (__builtin_expect(cont_sp > 0, 1)) {
          uint32_t nc  = cont_stack[cont_sp - 1];
          uint32_t nct = CONT_TAG_OF(nc);
          if (__builtin_expect(nct == CONT_TUP_DONE, 1)) {
            val = make_tup(CONT_VAL_OF(c), val);
            c = nc;
            continue;
          }
          // Deforestation: next is REC/TAIL_REC, skip last alloc
          if (nct == CONT_REC || nct == CONT_TAIL_REC) {
            cont_sp--;
            saved_sp = CONT_VAL_OF(nc);
            if (nct == CONT_TAIL_REC) {
              for (uint32_t i = saved_sp; i < sub_sp; i++)
                if (sub_stack[i] != 0) free_term(sub_stack[i]);
              sub_sp = saved_sp;
            }
            stat_get++;
            stat_mat++;
            uint16_t fst0 = CONT_VAL_OF(c);
            uint8_t  mat0 = cc[1];
            uint32_t d0   = mat0 - OP_MAT_BASE;
            uint32_t npc  = (TAG(fst0) == TAG_BT0) ? 2 : 2 + d0;
            uint32_t body_pc = feed_term(npc, val);
            if (cc[body_pc] == OP_REC) {
              stat_fun++;
              cont_stack[cont_sp++] = CONT_MAKE(CONT_TAIL_REC, saved_sp);
              pc = body_pc + 1;
            } else {
              cont_stack[cont_sp++] = CONT_MAKE(CONT_REC_GC, saved_sp);
              pc = body_pc;
            }
            goto eval;
          }
        }
        val = make_tup(CONT_VAL_OF(c), val);
        break;
      }
      goto ret;
    }

    // TUP_SND: rewrite in place to DONE
    if (CONT_TAG_OF(c) == CONT_TUP_SND) {
      cont_stack[cont_sp - 1] = CONT_MAKE(CONT_TUP_DONE, val);
      pc = CONT_VAL_OF(c);
      goto eval;
    }

    cont_sp--;
    switch (CONT_TAG_OF(c)) {
      case CONT_REC: {
        saved_sp = CONT_VAL_OF(c);
        goto post_feed;
      }
      case CONT_TAIL_REC: {
        saved_sp = CONT_VAL_OF(c);
        for (uint32_t i = saved_sp; i < sub_sp; i++)
          if (sub_stack[i] != 0) free_term(sub_stack[i]);
        sub_sp = saved_sp;
        goto post_feed;
      }
      case CONT_REC_GC: {
        uint32_t sp = CONT_VAL_OF(c);
        for (uint32_t i = sp; i < sub_sp; i++)
          if (sub_stack[i] != 0) free_term(sub_stack[i]);
        sub_sp = sp;
        goto ret;
      }
      default: __builtin_unreachable();
    }
  }

  // ---- POST_FEED: inline first GET+MAT, then feed_term ----
  post_feed: {
    stat_get++;
    uint32_t loc0 = LOC(val);
    uint16_t fst0 = heap[loc0];
    uint16_t snd0 = heap[loc0 + 1];
    free_pair(loc0);
    stat_mat++;
    uint8_t  mat0   = cc[1];
    uint32_t delta0 = mat0 - OP_MAT_BASE;
    uint32_t npc = (TAG(fst0) == TAG_BT0) ? 2 : 2 + delta0;
    uint32_t body_pc = feed_term(npc, snd0);
    if (cc[body_pc] == OP_REC) {
      stat_fun++;
      cont_stack[cont_sp++] = CONT_MAKE(CONT_TAIL_REC, saved_sp);
      pc = body_pc + 1;
    } else {
      cont_stack[cont_sp++] = CONT_MAKE(CONT_REC_GC, saved_sp);
      pc = body_pc;
    }
    goto eval;
  }
}

// ---- Printer ----
static void print_term(uint16_t p) {
  switch (TAG(p)) {
    case TAG_NUL: printf("()"); break;
    case TAG_BT0: printf("0");  break;
    case TAG_BT1: printf("1");  break;
    case TAG_TUP:
      printf("(");
      print_term(heap_fst(p));
      printf(",");
      print_term(heap_snd(p));
      printf(")");
      break;
  }
}

// ---- Main ----
int main(int argc, char **argv) {
  memset(heap, 0, sizeof(heap));

  // Program source (UTF-8 λ = \xCE\xBB)
  const char *prog_src =
    "\xCE\xBB! \xCE\xBB{\n"
    "  0: \xCE\xBB! \xCE\xBB{\n"
    "    0: \xCE\xBBx. ~(0,(1,(~(1,(1,x)),(0,()))))\n"
    "    1: \xCE\xBB! \xCE\xBB! \xCE\xBB{\n"
    "      0: \xCE\xBB(). \xCE\xBBzs. zs\n"
    "      1: \xCE\xBB! \xCE\xBB{\n"
    "        0: \xCE\xBBxs. \xCE\xBBzs. ~(0,(1,(xs,(1,(0,zs)))))\n"
    "        1: \xCE\xBBxs. \xCE\xBBzs. ~(0,(0,~(1,(0,(zs,(1,(1,xs)))))))\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  1: \xCE\xBB! \xCE\xBB{\n"
    "    0: \xCE\xBB! \xCE\xBB! \xCE\xBB{\n"
    "      0: \xCE\xBB(). \xCE\xBBys. ys\n"
    "      1: \xCE\xBB! \xCE\xBBx. \xCE\xBBxs. \xCE\xBBys. (1,(x,~(1,(0,(xs,ys)))))\n"
    "    }\n"
    "    1: \xCE\xBB! \xCE\xBB{\n"
    "      0: \xCE\xBB(). (0,())\n"
    "      1: \xCE\xBB! \xCE\xBB{\n"
    "        0: \xCE\xBBxs. (1,(1,~(1,(1,xs))))\n"
    "        1: \xCE\xBBxs. (1,(0,xs))\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}";

  // Compile program to bytecode
  {
    Src s = { prog_src, 0, (int)strlen(prog_src) };
    cname_depth = 0;
    code_len = 0;
    compile_func(&s);
  }

  printf("Bytecode (%u bytes): ", code_len);
  print_code(0, 0);
  printf("\n");

  // Print raw bytecode hex
  printf("Hex: ");
  for (uint32_t i = 0; i < code_len; i++) printf("%02x ", code[i]);
  printf("\n");

  // Build n = 40 one-bits (LSB-first): (1,(1,...(0,())))
  uint16_t n = PTR_NUL;
  n = make_tup(PTR_BT0, n); // (0,())
  for (int i = 0; i < 22; i++) {
    n = make_tup(PTR_BT1, make_tup(PTR_BT1, n)); // (1,(1,prev))
  }

  // arg = (0, (0, n)) for ~(0,(0,n)) = main(n)
  uint16_t inner = make_tup(PTR_BT0, n);
  uint16_t arg = make_tup(PTR_BT0, inner);

  printf("Input built. Starting evaluation...\n");
  fflush(stdout);

  struct timespec ts0, ts1;
  clock_gettime(CLOCK_MONOTONIC, &ts0);

  // Top-level call: feed arg into program, evaluate body
  stat_fun = 1;
  sub_sp = 0;
  uint32_t body_pc = feed_term(0, arg);
  fprintf(stderr, "DEBUG: after initial feed, sub_sp=%u, body_pc=%u\n", sub_sp, body_pc);

  uint16_t result = eval_iterative(body_pc);

  // GC top-level bindings
  for (uint32_t i = 0; i < sub_sp; i++) {
    if (sub_stack[i] != 0) free_term(sub_stack[i]);
  }
  sub_sp = 0;

  clock_gettime(CLOCK_MONOTONIC, &ts1);
  double elapsed = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  // Print result
  print_term(result);
  printf("\n");

  // Print stats
  uint64_t total = stat_fun + stat_lam + stat_mat + stat_get + stat_use;
  printf("Interactions: %llu\n", (unsigned long long)total);
  printf("- APP-FUN: %llu\n", (unsigned long long)stat_fun);
  printf("- APP-LAM: %llu\n", (unsigned long long)stat_lam);
  printf("- APP-MAT: %llu\n", (unsigned long long)stat_mat);
  printf("- APP-GET: %llu\n", (unsigned long long)stat_get);
  printf("- APP-USE: %llu\n", (unsigned long long)stat_use);
  printf("Time: %.4f seconds\n", elapsed);
  if (elapsed > 0)
    printf("Speed: %.2f million interactions/second\n", total / elapsed / 1e6);

  // Debug info
  printf("Max sub_sp: %u / %d\n", max_sub_sp, MAX_SUB);
  printf("Max cont_sp: %u / %d\n", max_cont_sp, MAX_CONT);

  return 0;
}
