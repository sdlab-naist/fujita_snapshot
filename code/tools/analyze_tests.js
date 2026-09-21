#!/usr/bin/env node
/*
 * analyze_tests.js — strict analysis of test code via a Babel AST (RQ1)
 *
 * Implements the method of paper Section 3.2.1 with an AST. It parses test files with the
 * JavaScript/TypeScript AST parser (@babel/parser) and counts:
 *   - test_cases     : total number of test()/it() calls (including .skip/.only/.each/.failing)
 *   - assertions     : total number of expect() calls
 *   - snapshot_tests : number of tests whose body contains toMatchSnapshot / toMatchInlineSnapshot
 *   - unit_tests     : number of tests that do not contain the above
 *
 * A snapshot matcher is detected by the property name of a MemberExpression such as
 * expect(...).toMatchSnapshot(). toBe/toEqual/toHaveBeenCalled etc. are treated as unit tests
 * (paper 2.3.4).
 *
 * Usage:
 *   node analyze_tests.js <file1> [file2 ...]            # individual files
 *   node analyze_tests.js --files-from list.txt          # newline-separated list of paths
 *   echo '<source>' | node analyze_tests.js --stdin      # a single source from stdin
 *
 * Output: JSON. With --files-from / multiple files:
 *   { "totals": {...}, "perFile": { "<path>": {...} }, "errors": {...} }
 * With a single --stdin it prints the per-file metrics directly.
 */
'use strict';

const fs = require('fs');

let parser, traverseMod;
try {
  parser = require('@babel/parser');
  traverseMod = require('@babel/traverse');
} catch (e) {
  console.error(JSON.stringify({
    fatal: 'missing-deps',
    message: 'run: cd tools && npm install (@babel/parser, @babel/traverse)',
  }));
  process.exit(2);
}
const traverse = traverseMod.default || traverseMod;

const SNAPSHOT_MATCHERS = new Set(['toMatchSnapshot', 'toMatchInlineSnapshot']);
const TEST_NAMES = new Set(['test', 'it']);

// Decide whether the given callee is test/it (or a qualifier like test.skip, it.each, ...)
// and return the base name ('test' | 'it'); null if it is neither.
function testBaseName(callee) {
  // simple call: test(...) / it(...)
  if (callee.type === 'Identifier' && TEST_NAMES.has(callee.name)) {
    return callee.name;
  }
  // qualified: test.skip(...), it.only(...), it.each(...)(...), test.failing(...)
  if (callee.type === 'MemberExpression') {
    // walk down to the deepest object (it.each([...]) becomes a further CallExpression)
    let obj = callee.object;
    while (obj && obj.type === 'MemberExpression') obj = obj.object;
    if (obj && obj.type === 'Identifier' && TEST_NAMES.has(obj.name)) {
      return obj.name;
    }
    // case where object is a CallExpression, e.g. it.each([...])`...`
    if (obj && obj.type === 'CallExpression') {
      return testBaseName(obj.callee);
    }
  }
  // tagged templates like it.each`...`(...) where callee itself is a CallExpression
  if (callee.type === 'CallExpression') {
    return testBaseName(callee.callee);
  }
  return null;
}

// whether the subtree of a node contains a snapshot matcher call
function subtreeHasSnapshot(node) {
  let found = false;
  traverse(
    node,
    {
      noScope: true,
      MemberExpression(path) {
        const prop = path.node.property;
        const name = prop && (prop.name || prop.value);
        if (name && SNAPSHOT_MATCHERS.has(name)) {
          found = true;
          path.stop();
        }
      },
    }
  );
  return found;
}

function parse(source) {
  return parser.parse(source, {
    sourceType: 'unambiguous',
    allowReturnOutsideFunction: true,
    errorRecovery: true,
    plugins: [
      'jsx',
      'typescript',
      'classProperties',
      'decorators-legacy',
      'optionalChaining',
      'nullishCoalescingOperator',
      'topLevelAwait',
      'importAssertions',
    ],
  });
}

function analyzeSource(source) {
  const ast = parse(source);
  let testCases = 0;
  let assertions = 0;
  let snapshotTests = 0;
  let unitTests = 0;

  traverse(ast, {
    CallExpression(path) {
      const callee = path.node.callee;

      // count expect(...)
      if (callee.type === 'Identifier' && callee.name === 'expect') {
        assertions += 1;
        return;
      }

      // count and classify test()/it()
      const base = testBaseName(callee);
      if (base) {
        // a real test has its body callback as the last argument.
        // do not count intermediate config calls (whose argument is not a function),
        // e.g. test.each([...]).
        const args = path.node.arguments;
        const body = args.length ? args[args.length - 1] : null;
        const isFn = body && (body.type === 'ArrowFunctionExpression' ||
                              body.type === 'FunctionExpression');
        if (!isFn) return;
        testCases += 1;
        const hasSnap = subtreeHasSnapshot(body);
        if (hasSnap) snapshotTests += 1;
        else unitTests += 1;
      }
    },
  });

  return {
    test_cases: testCases,
    assertions: assertions,
    snapshot_tests: snapshotTests,
    unit_tests: unitTests,
  };
}

function emptyMetrics() {
  return { test_cases: 0, assertions: 0, snapshot_tests: 0, unit_tests: 0 };
}

function addInto(a, b) {
  a.test_cases += b.test_cases;
  a.assertions += b.assertions;
  a.snapshot_tests += b.snapshot_tests;
  a.unit_tests += b.unit_tests;
}

function main() {
  const argv = process.argv.slice(2);

  if (argv.includes('--stdin')) {
    const source = fs.readFileSync(0, 'utf8');
    try {
      process.stdout.write(JSON.stringify(analyzeSource(source)));
    } catch (e) {
      process.stdout.write(JSON.stringify(emptyMetrics()));
    }
    return;
  }

  let files = [];
  const fromIdx = argv.indexOf('--files-from');
  if (fromIdx !== -1) {
    const listPath = argv[fromIdx + 1];
    files = fs
      .readFileSync(listPath, 'utf8')
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean);
  } else {
    files = argv.filter((a) => !a.startsWith('--'));
  }

  const totals = emptyMetrics();
  const perFile = {};
  const errors = {};
  for (const f of files) {
    try {
      const src = fs.readFileSync(f, 'utf8');
      const m = analyzeSource(src);
      perFile[f] = m;
      addInto(totals, m);
    } catch (e) {
      errors[f] = String((e && e.message) || e);
    }
  }
  process.stdout.write(JSON.stringify({ totals, perFile, errors }));
}

main();
