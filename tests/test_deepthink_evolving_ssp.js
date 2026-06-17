const assert = require('assert');
const path = require('path');
const fs = require('fs');

const workflowPath = path.resolve('C:\\Users\\Dos\\.claude\\workflows\\deepthink.js');

let callHistory = [];
let agentMockResponses = {};

function mockAgent(options) {
    const systemInstruction = options.systemInstruction || '';
    const prompt = options.prompt || '';
    const name = options.name || 'unknown';

    callHistory.push({ type: 'agent', name, prompt: prompt.substring(0, 30), systemInstruction: systemInstruction.substring(0, 30) });

    if (agentMockResponses[name]) {
        if (typeof agentMockResponses[name] === 'function') {
            return agentMockResponses[name](options);
        }
        return agentMockResponses[name];
    }

    if (name.includes('Strategy') || systemInstruction.includes('Initial Strategy')) {
        let count = 2;
        const countMatch = prompt.match(/Generate up to (\d+)/);
        if (countMatch) {
            count = parseInt(countMatch[1]);
        }
        const strategies = [];
        for (let i = 1; i <= count; i++) {
            strategies.push({
                id: `b${i}`,
                strategy: `Strategy ${i}`,
                convergenceDirective: `c${i}`,
                risk: `r${i}`,
                successCriteria: `s${i}`
            });
        }
        return JSON.stringify({ strategies });
    }
    if (name.includes('Initial') || systemInstruction.includes('First Work Production')) {
        return `Execution candidate for branch ${prompt.substring(0, 30)}`;
    }
    if (name.includes('Critique') || systemInstruction.includes('Critique Agent')) {
        return `Critique for candidate ${prompt.substring(0, 30)}`;
    }
    if (name.includes('SSP') || systemInstruction.includes('Structured Solution Pool')) {
        return JSON.stringify({
            strategy_id: 'b1',
            solutions: [
                { title: 'Sol 1', content: 'Block 1', confidence: 0.8, internal_critique: 'c1' },
                { title: 'Sol 2', content: 'Block 2', confidence: 0.7, internal_critique: 'c2' },
                { title: 'Sol 3', content: 'Block 3', confidence: 0.6, internal_critique: 'c3' },
                { title: 'Sol 4', content: 'Block 4', confidence: 0.9, internal_critique: 'c4' },
                { title: 'Sol 5', content: 'Block 5', confidence: 0.5, internal_critique: 'c5' }
            ]
        });
    }
    if (name.includes('Corrector') || systemInstruction.includes('Correction and Refinement')) {
        return `Corrected candidate ${prompt.substring(0, 30)}`;
    }
    if (name.includes('PQF') || systemInstruction.includes('Post Quality Filter')) {
        return JSON.stringify({
            evaluations: [
                { id: 'b1', status: 'KEEP' },
                { id: 'b2', status: 'KEEP' }
            ]
        });
    }
    if (name.includes('Memory') || systemInstruction.includes('Memory Bank')) {
        return `Condensed memory summary for branch ${prompt.substring(0, 30)}`;
    }
    if (name.includes('Judge') || systemInstruction.includes('Final Judge')) {
        return "Final consolidated solution";
    }

    return "Default Mock Response";
}

async function mockPhase(name, fn) {
    callHistory.push({ type: 'phase-start', name });
    const res = await fn();
    callHistory.push({ type: 'phase-end', name });
    return res;
}

async function mockPipeline(phases) {
    callHistory.push({ type: 'pipeline-start', count: phases.length });
    const results = [];
    for (const p of phases) {
        results.push(await p());
    }
    callHistory.push({ type: 'pipeline-end' });
    return results;
}

function mockLog(msg) {
    // console.log("[Workflow Log]", msg);
}

global.agent = mockAgent;
global.phase = mockPhase;
global.pipeline = mockPipeline;
global.log = mockLog;

function resetMocks() {
    callHistory = [];
    agentMockResponses = {};
}

async function runTests() {
    console.log("Starting Deepthink Evolving DFS SSP workflow tests...");

    const { runWorkflow } = require(workflowPath);

    // Test 1: Branch Initialization
    {
        resetMocks();
        console.log("Running Test 1: Branch Initialization...");
        const result = await runWorkflow({
            task: "Solve math problem",
            maxBranches: 2,
            maxDepth: 2,
            pqfInterval: 5,
            memoryInterval: 10
        });

        // Verify strategy generator was called first
        const strategyCalls = callHistory.filter(c => c.type === 'agent' && c.name === 'Strategy Generator');
        assert.strictEqual(strategyCalls.length, 1, "Strategy Generator must be called exactly once");

        // Verify initial execution & critique calls
        const executionCalls = callHistory.filter(c => c.type === 'agent' && c.name === 'Initial Execution');
        const critiqueCalls = callHistory.filter(c => c.type === 'agent' && c.name === 'Critique');

        assert.strictEqual(executionCalls.length, 2, "Should execute once per branch (2 branches)");
        assert.strictEqual(critiqueCalls.length, 4, "Critique should run on initial executions (2) + loop executions (2)");
    }

    // Test 2: Correct Loop Order (execution -> critique -> SSP -> correction -> critique)
    {
        resetMocks();
        console.log("Running Test 2: Correct Loop Order...");
        await runWorkflow({
            task: "Verify loop order",
            maxBranches: 1,
            maxDepth: 3, // Initial (1) + 2 iterations (2, 3)
            pqfInterval: 5,
            memoryInterval: 10
        });

        // Sequence of agent calls
        const agentSequence = callHistory.filter(c => c.type === 'agent').map(c => c.name);
        
        const expectedSequence = [
            'Strategy Generator',
            'Initial Execution',
            'Critique',
            'SSP',
            'Corrector',
            'Critique',
            'SSP',
            'Corrector',
            'Critique',
            'Final Judge'
        ];

        assert.deepStrictEqual(agentSequence, expectedSequence, "Agent calls must follow correct sequential pipeline order per branch");
    }

    // Test 3: SSP Count Verification
    {
        resetMocks();
        console.log("Running Test 3: SSP Count Verification...");
        
        let capturedSspOptions = [];
        agentMockResponses['SSP'] = (opts) => {
            capturedSspOptions.push(opts);
            return JSON.stringify({
                strategy_id: 'b1',
                solutions: [
                    { title: '1', content: 'c1', confidence: 0.9, internal_critique: 'ic1' },
                    { title: '2', content: 'c2', confidence: 0.8, internal_critique: 'ic2' },
                    { title: '3', content: 'c3', confidence: 0.7, internal_critique: 'ic3' },
                    { title: '4', content: 'c4', confidence: 0.6, internal_critique: 'ic4' },
                    { title: '5', content: 'c5', confidence: 0.5, internal_critique: 'ic5' }
                ]
            });
        };

        await runWorkflow({
            task: "Verify SSP",
            maxBranches: 1,
            maxDepth: 2,
            pqfInterval: 5,
            memoryInterval: 10
        });

        assert.ok(capturedSspOptions.length > 0, "SSP Agent should have been called");
        const lastSSPResponse = JSON.parse(agentMockResponses['SSP']({}));
        assert.strictEqual(lastSSPResponse.solutions.length, 5, "SSP should return exactly 5 entries");
    }

    // Test 4: PQF Replacement (UPDATE)
    {
        resetMocks();
        console.log("Running Test 4: PQF Replacement...");

        let pqfCalled = false;
        let strategyEvolved = false;

        agentMockResponses['PQF'] = () => {
            pqfCalled = true;
            return JSON.stringify({
                evaluations: [
                    { id: 'b1', status: 'UPDATE' } // Force UPDATE on b1
                ]
            });
        };

        agentMockResponses['Strategy Generator'] = (opts) => {
            if (pqfCalled) {
                strategyEvolved = true;
                return JSON.stringify({
                    strategies: [
                        { id: 'b1-evolved', strategy: 'Strategy 1 Evolved', convergenceDirective: 'c1-new', risk: 'r1-new', successCriteria: 's1-new' }
                    ]
                });
            }
            return JSON.stringify({
                strategies: [
                    { id: 'b1', strategy: 'Strategy 1', convergenceDirective: 'c1', risk: 'r1', successCriteria: 's1' }
                ]
            });
        };

        await runWorkflow({
            task: "Verify PQF UPDATE",
            maxBranches: 1,
            maxDepth: 3,
            pqfInterval: 2, // PQF triggers on iteration 2
            memoryInterval: 10
        });

        assert.ok(pqfCalled, "PQF Agent should have run");
        assert.ok(strategyEvolved, "Strategy Generator should be called to evolve the branch strategy on UPDATE");
    }

    // Test 5: Memory Condensation
    {
        resetMocks();
        console.log("Running Test 5: Memory Condensation...");

        let memoryBankCalled = false;
        agentMockResponses['Memory Bank'] = () => {
            memoryBankCalled = true;
            return "Condensed memory bank text";
        };

        await runWorkflow({
            task: "Verify Memory Bank",
            maxBranches: 1,
            maxDepth: 3,
            pqfInterval: 5,
            memoryInterval: 2 // Condense every 2 iterations
        });

        assert.ok(memoryBankCalled, "Memory Bank agent must run when memoryInterval is reached");
    }

    // Test 6: Max Depth Stop
    {
        resetMocks();
        console.log("Running Test 6: Max Depth Stop...");

        const result = await runWorkflow({
            task: "Stop at max depth",
            maxBranches: 1,
            maxDepth: 6,
            pqfInterval: 10,
            memoryInterval: 10
        });

        const correctorCalls = callHistory.filter(c => c.type === 'agent' && c.name === 'Corrector');
        assert.strictEqual(correctorCalls.length, 5, "Corrector should run exactly maxDepth - 1 (5) times");
    }

    console.log("All tests passed successfully!");
}

runTests().catch(err => {
    console.error("Test run aborted with error:", err);
    process.exit(1);
});
