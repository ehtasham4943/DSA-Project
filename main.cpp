// StructureCheck: automated C++ grader + AST-based plagiarism detector (single file)
// Build:  g++ -std=c++17 -O2 main.cpp -o structurecheck      (Linux / WSL)
// Usage:  ./structurecheck grade  <problem_dir> <submissions_dir>
//         ./structurecheck detect <submissions_dir> [--threshold 0.7] [--top 10] [--minsize 4] [--template file.cpp]
//         ./structurecheck all    <problem_dir> <submissions_dir> [detect options]
// Problem dir: 1.in/1.out, 2.in/2.out ... (+ optional weights.txt: "<case> <weight>").  Output: grades.csv, report.txt
#include <cstddef>
#include <functional>
#include <utility>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <cctype>
#include <initializer_list>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <iomanip>
#include <filesystem>

// ======================= containers.h =======================
// Own templated containers: Vector, Stack, HashMap (chaining), MaxHeap, mergeSort

namespace sc {

template <class T> class Vector {
    T* d = nullptr; size_t n = 0, cap = 0;
    void grow() {
        size_t nc = cap ? cap * 2 : 8; T* nd = new T[nc];
        for (size_t i = 0; i < n; i++) nd[i] = std::move(d[i]);
        delete[] d; d = nd; cap = nc;
    }
public:
    Vector() = default;
    Vector(const Vector& o) { for (size_t i = 0; i < o.n; i++) push_back(o.d[i]); }
    Vector(Vector&& o) noexcept : d(o.d), n(o.n), cap(o.cap) { o.d = nullptr; o.n = o.cap = 0; }
    Vector& operator=(Vector o) { swap(o); return *this; }
    ~Vector() { delete[] d; }
    void swap(Vector& o) { std::swap(d, o.d); std::swap(n, o.n); std::swap(cap, o.cap); }
    void push_back(const T& v) { T tmp = v; if (n == cap) grow(); d[n++] = std::move(tmp); }
    void pop_back() { --n; }
    T& operator[](size_t i) { return d[i]; }
    const T& operator[](size_t i) const { return d[i]; }
    T& back() { return d[n - 1]; }
    size_t size() const { return n; }
    bool empty() const { return n == 0; }
    void clear() { n = 0; }
    T* begin() { return d; }
    T* end() { return d + n; }
    const T* begin() const { return d; }
    const T* end() const { return d + n; }
};

template <class T> class Stack {
    Vector<T> v;
public:
    void push(const T& x) { v.push_back(x); }
    T pop() { T x = v.back(); v.pop_back(); return x; }
    T& top() { return v.back(); }
    bool empty() const { return v.empty(); }
    size_t size() const { return v.size(); }
};

template <class K, class V> class HashMap {
    struct Node { K key; V val; Node* next; };
    Node** b; size_t cap, n;
    size_t idx(const K& k) const { return std::hash<K>()(k) % cap; }
    void rehash() {
        size_t nc = cap * 2; Node** nb = new Node*[nc]();
        for (size_t i = 0; i < cap; i++) {
            Node* x = b[i];
            while (x) { Node* nx = x->next; size_t j = std::hash<K>()(x->key) % nc; x->next = nb[j]; nb[j] = x; x = nx; }
        }
        delete[] b; b = nb; cap = nc;
    }
public:
    explicit HashMap(size_t c = 64) : b(new Node*[c]()), cap(c), n(0) {}
    HashMap(const HashMap&) = delete;
    HashMap& operator=(const HashMap&) = delete;
    ~HashMap() {
        for (size_t i = 0; i < cap; i++) { Node* x = b[i]; while (x) { Node* nx = x->next; delete x; x = nx; } }
        delete[] b;
    }
    V* find(const K& k) { for (Node* x = b[idx(k)]; x; x = x->next) if (x->key == k) return &x->val; return nullptr; }
    bool contains(const K& k) { return find(k) != nullptr; }
    V& operator[](const K& k) {
        V* f = find(k); if (f) return *f;
        if (n + 1 > cap * 2) rehash();
        size_t i = idx(k); b[i] = new Node{k, V(), b[i]}; n++; return b[i]->val;
    }
    size_t size() const { return n; }
    template <class F> void forEach(F f) const {
        for (size_t i = 0; i < cap; i++) for (Node* x = b[i]; x; x = x->next) f(x->key, x->val);
    }
};

template <class T> class MaxHeap {   // uses operator<
    Vector<T> a;
public:
    void push(const T& x) {
        a.push_back(x); size_t i = a.size() - 1;
        while (i > 0) { size_t p = (i - 1) / 2; if (a[p] < a[i]) { std::swap(a[p], a[i]); i = p; } else break; }
    }
    T pop() {
        T top = a[0]; a[0] = a.back(); a.pop_back(); size_t i = 0, n = a.size();
        while (true) {
            size_t l = 2 * i + 1, r = l + 1, m = i;
            if (l < n && a[m] < a[l]) m = l;
            if (r < n && a[m] < a[r]) m = r;
            if (m == i) break;
            std::swap(a[m], a[i]); i = m;
        }
        return top;
    }
    bool empty() const { return a.empty(); }
    size_t size() const { return a.size(); }
};

template <class T, class Less> void mergeSortRec(Vector<T>& v, Vector<T>& tmp, size_t lo, size_t hi, Less less) {
    if (hi - lo < 2) return;
    size_t mid = (lo + hi) / 2;
    mergeSortRec(v, tmp, lo, mid, less); mergeSortRec(v, tmp, mid, hi, less);
    size_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi) tmp[k++] = less(v[j], v[i]) ? v[j++] : v[i++];
    while (i < mid) tmp[k++] = v[i++];
    while (j < hi) tmp[k++] = v[j++];
    for (size_t x = lo; x < hi; x++) v[x] = tmp[x];
}
template <class T, class Less> void mergeSort(Vector<T>& v, Less less) {
    Vector<T> tmp; for (size_t i = 0; i < v.size(); i++) tmp.push_back(v[i]);
    mergeSortRec(v, tmp, 0, v.size(), less);
}

} // namespace sc

// ======================= parser.h =======================
// Lexer + recursive-descent parser for a C++ subset -> normalized hashed AST

namespace sc {

enum TokType { T_ID, T_NUM, T_STR, T_CHR, T_OP, T_END };
struct Token { TokType type = T_END; std::string text; };

inline Vector<Token> lex(const std::string& s) {
    Vector<Token> raw; size_t i = 0, n = s.size(); bool lineStart = true;
    static const char* ops[] = {"<<=", ">>=", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
                                "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "::", nullptr};
    while (i < n) {
        char c = s[i];
        if (c == '\n') { lineStart = true; i++; continue; }
        if (isspace((unsigned char)c)) { i++; continue; }
        if (c == '#' && lineStart) { while (i < n && s[i] != '\n') { if (s[i] == '\\' && i + 1 < n) i++; i++; } continue; }
        lineStart = false;
        if (c == '/' && i + 1 < n && s[i + 1] == '/') { while (i < n && s[i] != '\n') i++; continue; }
        if (c == '/' && i + 1 < n && s[i + 1] == '*') { i += 2; while (i + 1 < n && !(s[i] == '*' && s[i + 1] == '/')) i++; i += 2; continue; }
        if (isalpha((unsigned char)c) || c == '_') {
            size_t j = i; while (j < n && (isalnum((unsigned char)s[j]) || s[j] == '_')) j++;
            Token t; t.type = T_ID; t.text = s.substr(i, j - i); raw.push_back(t); i = j; continue;
        }
        if (isdigit((unsigned char)c) || (c == '.' && i + 1 < n && isdigit((unsigned char)s[i + 1]))) {
            size_t j = i;
            while (j < n && (isalnum((unsigned char)s[j]) || s[j] == '.' ||
                   ((s[j] == '+' || s[j] == '-') && j > i && (s[j - 1] == 'e' || s[j - 1] == 'E')))) j++;
            Token t; t.type = T_NUM; t.text = s.substr(i, j - i); raw.push_back(t); i = j; continue;
        }
        if (c == '"' || c == '\'') {
            size_t j = i + 1; while (j < n && s[j] != c) { if (s[j] == '\\') j++; j++; }
            Token t; t.type = (c == '"') ? T_STR : T_CHR; raw.push_back(t); i = j + 1; continue;
        }
        std::string op(1, c);
        for (int k = 0; ops[k]; k++) { std::string o = ops[k]; if (s.compare(i, o.size(), o) == 0) { op = o; break; } }
        Token t; t.type = T_OP; t.text = op; raw.push_back(t); i += op.size();
    }
    // drop "std ::" so std::cout == cout
    Vector<Token> out;
    for (size_t k = 0; k < raw.size(); k++) {
        if (raw[k].type == T_ID && raw[k].text == "std" && k + 1 < raw.size() && raw[k + 1].text == "::") { k++; continue; }
        out.push_back(raw[k]);
    }
    out.push_back(Token());
    return out;
}

struct Node {
    std::string label; uint64_t hash = 0; int size = 1; Vector<Node*> kids;
    Node() = default;
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    // nodes may be shared (x+=y reuses x), so the tree is not freed recursively; memory is small and freed at exit
};

inline uint64_t hashStr(const std::string& s) { uint64_t h = 1469598103934665603ULL; for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; } return h; }
inline uint64_t mixHash(uint64_t h, uint64_t x) { h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); return h * 1099511628211ULL; }

inline void finalize(Node* n) {  // recursion: hash from label + children's hashes
    n->hash = hashStr(n->label); n->size = 1;
    for (Node* k : n->kids) { n->hash = mixHash(n->hash, k->hash); n->size += k->size; }
}
inline Node* mkv(const std::string& l, const Vector<Node*>& k) { Node* n = new Node; n->label = l; n->kids = k; finalize(n); return n; }
inline Node* mk(const std::string& l, std::initializer_list<Node*> k = {}) { Vector<Node*> v; for (Node* x : k) v.push_back(x); return mkv(l, v); }

class Parser {
    Vector<Token> t; size_t p = 0; Vector<std::string> userTypes;
    Token& cur() { return t[p]; }
    bool isOp(const char* s) { return cur().type == T_OP && cur().text == s; }
    bool isId(const char* s) { return cur().type == T_ID && cur().text == s; }
    bool accept(const char* s) { if (isOp(s)) { p++; return true; } return false; }
    [[noreturn]] void fail(const std::string& m) { throw std::runtime_error("unsupported syntax (" + m + ") near '" + cur().text + "' token #" + std::to_string(p)); }
    void expect(const char* s) { if (!accept(s)) fail(std::string("expected '") + s + "'"); }

    static bool inList(const std::string& s, const char* const* L) { for (int i = 0; L[i]; i++) if (s == L[i]) return true; return false; }
    bool isUserType(const std::string& s) { for (auto& u : userTypes) if (u == s) return true; return false; }
    static const char* const* typeKw() {
        static const char* kw[] = {"int","long","short","char","bool","float","double","void","auto","unsigned","signed","const","static",
                                   "string","vector","size_t","map","set","queue","stack","pair","unordered_map","unordered_set",nullptr};
        return kw;
    }
    bool isTypeName(const std::string& s) { return inList(s, typeKw()) || isUserType(s); }
    bool isStreamName(const std::string& s) {
        return s == "cout" || s == "cin" || s == "cerr" || s == "endl" || s == "NULL" || s == "nullptr";
    }
    bool isStdName(const std::string& s) {   // std function / method names (only kept when used as calls)
        static const char* nm[] = {"cout","cin","cerr","endl","printf","scanf","puts","getline","sort","min","max","abs","swap","sqrt","pow",
                                   "sizeof","push_back","pop_back","size","empty","begin","end","front","back","top","pop","push","insert",
                                   "erase","find","count","clear","reverse","length","substr","main","NULL","nullptr",nullptr};
        return inList(s, nm);
    }
    bool isDeclStart() {
        if (cur().type != T_ID) return false;
        const std::string& s = cur().text;
        if (inList(s, typeKw())) return true;
        if (isUserType(s)) { Token& n = t[p + 1]; return n.type == T_ID || n.text == "*" || n.text == "&" || n.text == "<"; }
        return false;
    }
    Node* leaf(const std::string& l) { return mk(l); }
    Node* wrapBlock(Node* s) {
        if (s->label == "splice") { s->label = "block"; finalize(s); return s; }
        return s->label == "block" ? s : mk("block", {s});
    }

    Node* parseType() {
        std::string lab = "type:"; bool any = false;
        while (cur().type == T_ID && isTypeName(cur().text)) { lab += isUserType(cur().text) ? "UT" : cur().text; lab += ' '; p++; any = true; }
        if (!any) fail("expected type");
        if (isOp("<")) {
            int depth = 0;
            do {
                if (isOp("<")) depth++; else if (isOp(">")) depth--; else if (isOp(">>")) depth -= 2;
                else if (cur().type == T_END) fail("unterminated template");
                else if (cur().type == T_ID) lab += isUserType(cur().text) ? "UT" : cur().text;
                p++;
            } while (depth > 0);
            lab += "<>";
        }
        while (isOp("*") || isOp("&") || isOp("&&") || isId("const")) { lab += cur().text; p++; }
        return leaf(lab);
    }
    Node* parseInitializer() {
        if (accept("{")) {
            Vector<Node*> k;
            while (!isOp("}")) { k.push_back(parseExpr()); if (!accept(",")) break; }
            expect("}"); return mkv("initlist", k);
        }
        return parseExpr();
    }
    Node* parseDeclarators(Node* ty) {
        Vector<Node*> kids; kids.push_back(ty);
        do {
            while (accept("*") || accept("&")) {}
            if (cur().type != T_ID) fail("expected name");
            p++;
            Vector<Node*> dk;
            while (accept("[")) { if (!isOp("]")) dk.push_back(parseExpr()); expect("]"); dk.push_back(leaf("dim")); }
            if (accept("=")) dk.push_back(parseInitializer());
            else if (isOp("{")) dk.push_back(parseInitializer());
            else if (accept("(")) {
                Vector<Node*> a; while (!isOp(")")) { a.push_back(parseExpr()); if (!accept(",")) break; }
                expect(")"); dk.push_back(mkv("ctor", a));
            }
            kids.push_back(mkv("declarator", dk));
        } while (accept(","));
        return mkv("decl", kids);
    }

    // ---- expressions ----
    static int prec(const std::string& o) {
        if (o == "*" || o == "/" || o == "%") return 13;
        if (o == "+" || o == "-") return 12;
        if (o == "<<" || o == ">>") return 11;
        if (o == "<" || o == "<=" || o == ">" || o == ">=") return 10;
        if (o == "==" || o == "!=") return 9;
        if (o == "&") return 8;
        if (o == "^") return 7;
        if (o == "|") return 6;
        if (o == "&&") return 5;
        if (o == "||") return 4;
        return 0;
    }
    Node* makeBin(std::string op, Node* a, Node* b) {
        if (op == ">") { op = "<"; std::swap(a, b); }           // a>b == b<a
        else if (op == ">=") { op = "<="; std::swap(a, b); }
        bool comm = op == "+" || op == "*" || op == "==" || op == "!=" || op == "&&" || op == "||" || op == "&" || op == "|" || op == "^";
        if (comm && b->hash < a->hash) std::swap(a, b);          // a+b == b+a
        return mk("bin:" + op, {a, b});
    }
    Node* parseBinary() {   // shunting-yard: operand stack + operator stack
        Stack<Node*> vals; Stack<std::string> ops;
        auto reduce = [&]() { std::string o = ops.pop(); Node* b = vals.pop(); Node* a = vals.pop(); vals.push(makeBin(o, a, b)); };
        vals.push(parseUnary());
        while (cur().type == T_OP && prec(cur().text) > 0) {
            std::string op = cur().text; p++;
            while (!ops.empty() && prec(ops.top()) >= prec(op)) reduce();
            ops.push(op); vals.push(parseUnary());
        }
        while (!ops.empty()) reduce();
        return vals.pop();
    }
    Node* parseTernary() {
        Node* c = parseBinary();
        if (accept("?")) { Node* a = parseExpr(); expect(":"); Node* b = parseTernary(); return mk("?:", {c, a, b}); }
        return c;
    }
    Node* parseExpr() {
        Node* l = parseTernary();
        static const char* as[] = {"=","+=","-=","*=","/=","%=","&=","|=","^=","<<=",">>=",nullptr};
        if (cur().type == T_OP && inList(cur().text, as)) {
            std::string op = cur().text; p++;
            Node* r = parseExpr();
            if (op == "+=" || op == "-=" || op == "*=" || op == "/=" || op == "%=")   // x+=y == x=x+y (structure only)
                return mk("assign", {l, makeBin(op.substr(0, 1), l, r)});
            return mk(op == "=" ? "assign" : "assign:" + op, {l, r});
        }
        return l;
    }
    Node* parseUnary() {
        if (cur().type == T_OP) {
            const std::string o = cur().text;
            if (o == "!" || o == "-" || o == "+" || o == "~" || o == "*" || o == "&" || o == "++" || o == "--") { p++; return mk("un:" + o, {parseUnary()}); }
        }
        if (isId("new") || isId("delete")) fail("new/delete");
        return parsePostfix(parsePrimary());
    }
    Node* parsePrimary() {
        Token& k = cur();
        if (k.type == T_NUM) { p++; bool fl = k.text.find('.') != std::string::npos; return leaf(fl ? "num:float" : "num:int"); }
        if (k.type == T_STR) { p++; return leaf("str"); }
        if (k.type == T_CHR) { p++; return leaf("char"); }
        if (k.type == T_ID) {
            std::string s = k.text; p++;
            if (s == "true" || s == "false") return leaf("bool");
            if (isOp("(")) return leaf(isStdName(s) ? s : "fn");
            return leaf(isStreamName(s) ? s : "var");
        }
        if (isOp("(")) {
            if (t[p + 1].type == T_ID && isTypeName(t[p + 1].text)) {   // C-style cast
                p++; Node* ty = parseType(); expect(")"); return mk("cast:" + ty->label, {parseUnary()});
            }
            p++; Node* e = parseExpr(); expect(")"); return e;
        }
        fail("expression");
    }
    Node* parsePostfix(Node* b) {
        while (true) {
            if (accept("(")) {
                Vector<Node*> a; a.push_back(b);
                while (!isOp(")")) { a.push_back(parseExpr()); if (!accept(",")) break; }
                expect(")"); b = mkv("call", a);
            } else if (accept("[")) { Node* i = parseExpr(); expect("]"); b = mk("index", {b, i}); }
            else if (isOp(".") || isOp("->")) {
                p++; if (cur().type != T_ID) fail("member");
                std::string m = cur().text; p++;
                b = mk(isStdName(m) && isOp("(") ? "member:" + m : "member", {b});
            } else if (isOp("++") || isOp("--")) { b = mk("post:" + cur().text, {b}); p++; }
            else break;
        }
        return b;
    }

    // ---- statements ----
    Node* parseBlock() {
        expect("{"); Vector<Node*> k;
        while (!isOp("}")) {
            if (cur().type == T_END) fail("missing '}'");
            Node* st = parseStmt();
            if (st->label == "splice") { for (Node* c : st->kids) k.push_back(c); } else k.push_back(st);
        }
        p++; return mkv("block", k);
    }
    Node* parseStmt() {
        if (isOp("{")) return parseBlock();
        if (accept(";")) return leaf("empty");
        if (isId("if")) {
            p++; expect("("); Node* c = parseExpr(); expect(")"); Node* th = wrapBlock(parseStmt());
            if (isId("else")) { p++; Node* el = wrapBlock(parseStmt()); return mk("if", {c, th, el}); }
            return mk("if", {c, th});
        }
        if (isId("while")) { p++; expect("("); Node* c = parseExpr(); expect(")"); return mk("while", {c, wrapBlock(parseStmt())}); }
        if (isId("do")) {
            p++; Node* b = wrapBlock(parseStmt());
            if (!isId("while")) fail("expected while");
            p++; expect("("); Node* c = parseExpr(); expect(")"); expect(";");
            return mk("dowhile", {b, c});
        }
        if (isId("for")) return parseFor();
        if (isId("return")) { p++; if (accept(";")) return leaf("return"); Node* e = parseExpr(); expect(";"); return mk("return", {e}); }
        if (isId("break")) { p++; expect(";"); return leaf("break"); }
        if (isId("continue")) { p++; expect(";"); return leaf("continue"); }
        if (isId("switch")) { p++; expect("("); Node* e = parseExpr(); expect(")"); return mk("switch", {e, parseBlock()}); }
        if (isId("case")) { p++; Node* e = parseTernary(); expect(":"); return mk("case", {e}); }
        if (isId("default")) { p++; expect(":"); return leaf("default"); }
        if (isDeclStart()) { Node* ty = parseType(); Node* d = parseDeclarators(ty); expect(";"); return d; }
        Node* e = parseExpr(); expect(";"); return mk("expr", {e});
    }
    Node* parseFor() {
        p++; expect("("); Node* init;
        if (accept(";")) init = leaf("empty");
        else if (isDeclStart()) {
            Node* ty = parseType();
            if (cur().type == T_ID && t[p + 1].type == T_OP && t[p + 1].text == ":") {
                p += 2; Node* r = parseExpr(); expect(")"); return mk("rangefor", {ty, r, wrapBlock(parseStmt())});
            }
            init = parseDeclarators(ty); expect(";");
        } else { init = mk("expr", {parseExpr()}); expect(";"); }
        Node* cond = isOp(";") ? leaf("bool") : parseExpr(); expect(";");
        Vector<Node*> incs;
        if (!isOp(")")) { do incs.push_back(mk("expr", {parseExpr()})); while (accept(",")); }
        expect(")");
        Node* body = wrapBlock(parseStmt());
        for (Node* n : incs) body->kids.push_back(n);
        finalize(body);
        return mk("splice", {init, mk("while", {cond, body})});   // for  ==>  init; while(cond){body; incr}
    }

    // ---- top level ----
    void skipToSemi() { while (!isOp(";") && cur().type != T_END) p++; p++; }
    Node* parseTop() {
        if (isId("using")) {
            if (t[p + 1].type == T_ID && t[p + 2].text == "=") userTypes.push_back(t[p + 1].text);
            skipToSemi(); return nullptr;
        }
        if (isId("typedef")) {
            size_t q = p; while (!(t[q].text == ";" && t[q].type == T_OP) && t[q].type != T_END) q++;
            if (q > 0 && t[q - 1].type == T_ID) userTypes.push_back(t[q - 1].text);
            p = q + 1; return nullptr;
        }
        if (isId("template") || isId("namespace") || isId("enum") || isId("union")) fail("keyword " + cur().text);
        if (isId("struct") || isId("class")) {
            p++; if (cur().type != T_ID) fail("struct name");
            size_t nameIdx = p; userTypes.push_back(cur().text); p++;
            if (isOp("{")) {
                p++; Vector<Node*> m;
                while (!isOp("}")) {
                    if (cur().type == T_END) fail("missing '}'");
                    if (isId("public") || isId("private") || isId("protected")) { p++; expect(":"); continue; }
                    Node* x = parseTop(); if (x) m.push_back(x);
                }
                p++; while (!isOp(";") && cur().type != T_END) p++;
                p++; return mkv("struct", m);
            }
            p = nameIdx;   // "struct X var;" -> ordinary declaration using the type name
        }
        Node* ty = parseType(); size_t save = p;
        if (cur().type != T_ID) fail("expected name");
        p++;
        if (isOp("(")) {
            p++; Vector<Node*> k; k.push_back(ty);
            while (!isOp(")")) {
                Node* pt = parseType(); if (cur().type == T_ID) p++;
                while (accept("[")) { if (!isOp("]")) parseExpr(); expect("]"); }
                if (accept("=")) parseExpr();
                k.push_back(mk("param", {pt})); if (!accept(",")) break;
            }
            expect(")");
            if (isOp("{")) k.push_back(parseBlock()); else expect(";");
            return mkv("func", k);
        }
        p = save; Node* d = parseDeclarators(ty); expect(";"); return d;
    }
public:
    explicit Parser(const std::string& src) : t(lex(src)) {}
    Node* parseProgram() {
        Vector<Node*> items;
        while (cur().type != T_END) { Node* x = parseTop(); if (x) items.push_back(x); }
        return mkv("program", items);
    }
};

} // namespace sc

// ======================= detector.h =======================
// AST-based plagiarism detector: subtree hashing, hash-table candidate matching, DFS clusters, top-K heap

namespace sc {

struct Profile {
    std::string name; int total = 0; bool ok = false; std::string error;
    HashMap<uint64_t, int> counts;       // subtree hash -> multiplicity
};

struct PairScore {
    int a, b; double s;
    bool operator<(const PairScore& o) const { return s < o.s; }
};

inline std::string readFile(const std::string& path) {
    std::ifstream f(path); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

class Detector {
    int minSize;
    HashMap<uint64_t, int> templ;        // hashes from instructor template: ignored
public:
    Vector<Profile*> profiles;
    explicit Detector(int minSubtree = 4) : minSize(minSubtree) {}
    ~Detector() { for (Profile* p : profiles) delete p; }

    void collect(Node* n, Profile* pr, bool isTemplate) {          // O(n) walk
        for (Node* k : n->kids) collect(k, pr, isTemplate);
        if (n->size < minSize) return;
        if (isTemplate) { templ[n->hash] = 1; return; }
        if (templ.contains(n->hash)) return;
        pr->counts[n->hash]++; pr->total++;
    }
    void setTemplate(const std::string& path) {
        try { Parser ps(readFile(path)); Node* r = ps.parseProgram(); Profile dummy; collect(r, &dummy, true); }
        catch (std::exception& e) { std::cerr << "template ignored: " << e.what() << "\n"; }
    }
    void addSubmission(const std::string& name, const std::string& path) {
        Profile* pr = new Profile; pr->name = name;
        try { Parser ps(readFile(path)); Node* r = ps.parseProgram(); collect(r, pr, false); pr->ok = true; }
        catch (std::exception& e) { pr->error = e.what(); }
        profiles.push_back(pr);
    }
    // S = 2|M| / (|T1| + |T2|), |M| = multiset intersection of subtree hashes
    double similarity(Profile* x, Profile* y) {
        if (x->total + y->total == 0) return 0.0;
        Profile* sm = x->counts.size() <= y->counts.size() ? x : y; Profile* lg = (sm == x) ? y : x;
        long m = 0;
        sm->counts.forEach([&](const uint64_t& h, const int& c) { int* o = lg->counts.find(h); if (o) m += (c < *o ? c : *o); });
        return 2.0 * m / (x->total + y->total);
    }
    // all candidate pairs (those sharing at least one hash) with their similarity
    Vector<PairScore> allPairs() {
        int N = (int)profiles.size();
        HashMap<uint64_t, Vector<int>> index;                       // hash table with chaining
        for (int i = 0; i < N; i++) {
            if (!profiles[i]->ok) continue;
            profiles[i]->counts.forEach([&](const uint64_t& h, const int&) { index[h].push_back(i); });
        }
        HashMap<uint64_t, int> seen; Vector<PairScore> out;
        index.forEach([&](const uint64_t&, const Vector<int>& L) {
            for (size_t x = 0; x < L.size(); x++) for (size_t y = x + 1; y < L.size(); y++) {
                int a = L[x], b = L[y]; if (a > b) std::swap(a, b);
                uint64_t key = (uint64_t)a * N + b;
                if (seen.contains(key)) continue;
                seen[key] = 1; out.push_back({a, b, similarity(profiles[a], profiles[b])});
            }
        });
        return out;
    }
    Vector<PairScore> topK(const Vector<PairScore>& pairs, int k) {  // max-heap
        MaxHeap<PairScore> h; for (const PairScore& p : pairs) h.push(p);
        Vector<PairScore> out; while (!h.empty() && (int)out.size() < k) out.push_back(h.pop());
        return out;
    }
    Vector<Vector<int>> clusters(const Vector<PairScore>& pairs, double threshold) {   // DFS on similarity graph
        int N = (int)profiles.size(); Vector<Vector<int>> adj; for (int i = 0; i < N; i++) adj.push_back(Vector<int>());
        for (const PairScore& p : pairs) if (p.s >= threshold) { adj[p.a].push_back(p.b); adj[p.b].push_back(p.a); }
        Vector<char> vis; for (int i = 0; i < N; i++) vis.push_back(0);
        Vector<Vector<int>> groups;
        for (int s = 0; s < N; s++) {
            if (vis[s]) continue;
            Vector<int> comp; Stack<int> st; st.push(s); vis[s] = 1;
            while (!st.empty()) {
                int u = st.pop(); comp.push_back(u);
                for (int v : adj[u]) if (!vis[v]) { vis[v] = 1; st.push(v); }
            }
            if (comp.size() >= 2) groups.push_back(comp);
        }
        return groups;
    }
};

} // namespace sc

// ======================= grader.h =======================
// Auto-grader: compile, run each test in a limited child process (fork/exec/setrlimit), compare output

namespace sc {

struct TestCase { std::string in, out; double weight; };
struct GradeResult { std::string student; bool compiled = false; int passed = 0, total = 0; double score = 0; std::string note; };

inline bool fileExists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0; }

inline Vector<TestCase> loadCases(const std::string& dir) {
    Vector<TestCase> cs; HashMap<int, double> w;
    std::ifstream wf(dir + "/weights.txt"); int id; double x; while (wf >> id >> x) w[id] = x;
    for (int i = 1; fileExists(dir + "/" + std::to_string(i) + ".in") && fileExists(dir + "/" + std::to_string(i) + ".out"); i++) {
        double* f = w.find(i);
        cs.push_back({dir + "/" + std::to_string(i) + ".in", dir + "/" + std::to_string(i) + ".out", f ? *f : 1.0});
    }
    return cs;
}

inline std::string normalizeOutput(const std::string& s) {      // trim trailing spaces per line + trailing blank lines
    std::istringstream in(s); std::string line, out; Vector<std::string> lines;
    while (std::getline(in, line)) { while (!line.empty() && isspace((unsigned char)line.back())) line.pop_back(); lines.push_back(line); }
    while (!lines.empty() && lines.back().empty()) lines.pop_back();
    for (auto& l : lines) out += l + "\n";
    return out;
}
inline std::string slurp(const std::string& p) { std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str(); }

enum RunStatus { RUN_OK, RUN_TLE, RUN_RE };

inline RunStatus runLimited(const std::string& bin, const std::string& in, const std::string& out, int cpuSec = 2, size_t memMB = 256, int wallMs = 4000) {
    pid_t pid = fork();
    if (pid == 0) {
        int fi = open(in.c_str(), O_RDONLY), fo = open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644), fe = open("/dev/null", O_WRONLY);
        dup2(fi, 0); dup2(fo, 1); dup2(fe, 2);
        rlimit r; r.rlim_cur = cpuSec; r.rlim_max = cpuSec + 1; setrlimit(RLIMIT_CPU, &r);
        r.rlim_cur = r.rlim_max = memMB * 1024 * 1024; setrlimit(RLIMIT_AS, &r);
        r.rlim_cur = r.rlim_max = 10 * 1024 * 1024; setrlimit(RLIMIT_FSIZE, &r);
        execl(bin.c_str(), bin.c_str(), (char*)nullptr); _exit(127);
    }
    if (pid < 0) return RUN_RE;
    auto start = std::chrono::steady_clock::now(); int st = 0;
    while (true) {
        pid_t w = waitpid(pid, &st, WNOHANG);
        if (w == pid) break;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        if (ms > wallMs) { kill(pid, SIGKILL); waitpid(pid, &st, 0); return RUN_TLE; }
        usleep(2000);
    }
    if (WIFSIGNALED(st) && WTERMSIG(st) == SIGXCPU) return RUN_TLE;
    if (WIFEXITED(st) && WEXITSTATUS(st) == 0) return RUN_OK;
    return RUN_RE;
}

inline GradeResult gradeSubmission(const std::string& student, const std::string& src, const Vector<TestCase>& cases, const std::string& tmp) {
    GradeResult g; g.student = student; g.total = (int)cases.size();
    std::string bin = tmp + "/" + student + ".bin", log = tmp + "/" + student + ".log", outp = tmp + "/" + student + ".out";
    std::string cmd = "timeout 30 g++ -std=c++17 -O2 \"" + src + "\" -o \"" + bin + "\" 2> \"" + log + "\"";
    g.compiled = (std::system(cmd.c_str()) == 0);
    if (!g.compiled) { g.note = "compile error"; return g; }
    double got = 0, all = 0; int tle = 0, re = 0, wa = 0;
    for (const TestCase& c : cases) {
        all += c.weight;
        RunStatus rs = runLimited(bin, c.in, outp);
        if (rs == RUN_TLE) { tle++; continue; }
        if (rs == RUN_RE) { re++; continue; }
        if (normalizeOutput(slurp(outp)) == normalizeOutput(slurp(c.out))) { got += c.weight; g.passed++; } else wa++;
    }
    g.score = all > 0 ? 100.0 * got / all : 0;
    g.note = "WA=" + std::to_string(wa) + " TLE=" + std::to_string(tle) + " RE=" + std::to_string(re);
    return g;
}

} // namespace sc

// ======================= main.cpp =======================
// StructureCheck: automated C++ grader + AST-based plagiarism detector

using namespace sc;
namespace fs = std::filesystem;

struct Sub { std::string name, path; };

static Vector<Sub> listSubmissions(const std::string& dir) {
    Vector<Sub> v;
    for (auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".cpp") v.push_back({e.path().stem().string(), e.path().string()});
    mergeSort(v, [](const Sub& a, const Sub& b) { return a.name < b.name; });
    return v;
}

static void usage() {
    std::cout << "StructureCheck\n"
        "  structurecheck grade  <problem_dir> <submissions_dir>\n"
        "  structurecheck detect <submissions_dir> [--threshold 0.7] [--top 10] [--minsize 4] [--template file.cpp]\n"
        "  structurecheck all    <problem_dir> <submissions_dir> [detect options]\n"
        "Problem dir holds 1.in/1.out, 2.in/2.out ... and optional weights.txt (lines: <case> <weight>).\n"
        "Outputs: grades.csv and report.txt\n";
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(); return 1; }
    std::string mode = argv[1], probDir, subDir, templ; double threshold = 0.7; int topK = 10, minSize = 4;
    int a = 2;
    if (mode == "grade" || mode == "all") { if (argc < 4) { usage(); return 1; } probDir = argv[2]; subDir = argv[3]; a = 4; }
    else if (mode == "detect") { subDir = argv[2]; a = 3; }
    else { usage(); return 1; }
    for (; a + 1 < argc; a += 2) {
        std::string k = argv[a];
        if (k == "--threshold") threshold = std::stod(argv[a + 1]);
        else if (k == "--top") topK = std::stoi(argv[a + 1]);
        else if (k == "--minsize") minSize = std::stoi(argv[a + 1]);
        else if (k == "--template") templ = argv[a + 1];
    }
    if (!fs::is_directory(subDir)) { std::cerr << "Not a directory: " << subDir << "\n"; return 1; }
    Vector<Sub> subs = listSubmissions(subDir);
    std::ostringstream rep; rep << std::fixed << std::setprecision(1);
    rep << "===== StructureCheck report =====\nSubmissions: " << subs.size() << "\n\n";

    if (mode == "grade" || mode == "all") {
        Vector<TestCase> cases = loadCases(probDir);
        if (cases.empty()) { std::cerr << "No test cases found in " << probDir << "\n"; return 1; }
        fs::create_directories(".sc_tmp");
        auto t0 = std::chrono::steady_clock::now();
        Vector<GradeResult> res;
        for (const Sub& s : subs) { res.push_back(gradeSubmission(s.name, s.path, cases, ".sc_tmp")); std::cerr << "graded " << s.name << "\n"; }
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        mergeSort(res, [](const GradeResult& x, const GradeResult& y) { return x.score > y.score; });   // best first
        std::ofstream csv("grades.csv"); csv << "student,compiled,passed,total,score,note\n";
        rep << "--- Grades (" << cases.size() << " test cases) ---\n";
        for (const GradeResult& g : res) {
            csv << g.student << "," << (g.compiled ? "yes" : "no") << "," << g.passed << "," << g.total << "," << g.score << "," << g.note << "\n";
            rep << std::left << std::setw(16) << g.student << std::right << std::setw(6) << g.score << "%  (" << g.passed << "/" << g.total << ")  " << g.note << "\n";
        }
        rep << "Grading time: " << std::setprecision(2) << secs << " s\n\n" << std::setprecision(1);
        fs::remove_all(".sc_tmp");
    }

    if (mode == "detect" || mode == "all") {
        auto t0 = std::chrono::steady_clock::now();
        Detector det(minSize); if (!templ.empty()) det.setTemplate(templ);
        for (const Sub& s : subs) det.addSubmission(s.name, s.path);
        rep << "--- Unsupported / skipped files ---\n"; int bad = 0;
        for (Profile* p : det.profiles) if (!p->ok) { rep << p->name << ": " << p->error << "\n"; bad++; }
        if (!bad) rep << "(none)\n";
        Vector<PairScore> pairs = det.allPairs();
        Vector<PairScore> top = det.topK(pairs, topK);
        rep << "\n--- Top " << topK << " similar pairs (evidence only; instructor decides) ---\n";
        for (const PairScore& p : top) rep << std::left << std::setw(14) << det.profiles[p.a]->name << " <-> " << std::setw(14) << det.profiles[p.b]->name
                                          << std::right << std::setw(6) << 100 * p.s << "%" << (p.s >= threshold ? "  ** FLAGGED" : "") << "\n";
        if (top.empty()) rep << "(no similar pairs)\n";
        Vector<Vector<int>> groups = det.clusters(pairs, threshold);
        rep << "\n--- Clusters (similarity >= " << 100 * threshold << "%) ---\n";
        int n = 1; for (auto& g : groups) { rep << "Cluster " << n++ << ": "; for (size_t i = 0; i < g.size(); i++) rep << (i ? ", " : "") << det.profiles[g[i]]->name; rep << "\n"; }
        if (groups.empty()) rep << "(none)\n";
        rep << "Detection time: " << std::setprecision(3) << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
    }
    std::cout << rep.str(); std::ofstream("report.txt") << rep.str();
    return 0;
}
