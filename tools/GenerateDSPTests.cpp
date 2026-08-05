// ------------------------------------------------------------------
// A standalone C++23 tool to auto-generate both exhaustive and 
// randomized deeply nested MLIR canonicalization tests.
// ------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <execution>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// ------------------------------------------------------------------
// Define an enumeration class for DSP operations
// ------------------------------------------------------------------
enum class DspOp
{
    DCT = 0,
    IDCT = 1
};

// ------------------------------------------------------------------
// Convert the enumeration to its corresponding MLIR string representation
// ------------------------------------------------------------------
constexpr std::string_view getOpName(const DspOp op)
{
    if (op == DspOp::DCT)
    {
        return "dsp.dct";
    }
    return "dsp.idct";
}

// ------------------------------------------------------------------
// Iteratively calculate the Binomial Coefficient C(n, k) 
// Provides overflow protection for extreme depths.
// ------------------------------------------------------------------
constexpr std::size_t calculateBinomialCoefficient(
    const std::size_t n, 
    const std::size_t k)
{
    // C(62, 31) is close to the limit of a 64-bit unsigned integer.
    // Return max value directly to avoid integer overflow.
    if (n > 62uz)
    {
        return std::numeric_limits<std::size_t>::max();
    }

    std::size_t result{1uz};
    for (std::size_t i = 1uz; i <= k; ++i)
    {
        result = result * (n - i + 1uz) / i;
    }
    return result;
}

// ------------------------------------------------------------------
// Default shuffle operation functor used as a strategy parameter.
// Encapsulates the standard Fisher-Yates shuffle algorithm.
// ------------------------------------------------------------------
struct DefaultShuffleOp
{
    template <typename RangeT, typename UrbgT>
        requires std::ranges::random_access_range<std::remove_cvref_t<RangeT>>
              && std::uniform_random_bit_generator<std::remove_cvref_t<UrbgT>>
    constexpr void operator()(RangeT&& range, UrbgT&& urbg) const
    {
        std::ranges::shuffle(
            std::forward<RangeT>(range), 
            std::forward<UrbgT>(urbg)
        );
    }
};

// ------------------------------------------------------------------
// Create the initial balanced pattern of length depth * 2.
// Uses if constexpr to dispatch between high-performance path 
// and the safe sequential fallback for complex objects.
// ------------------------------------------------------------------
template <
    typename InnerContainerT,
    typename BaseT,
    typename ExecutionPolicy>
requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
InnerContainerT createInitialPattern(
    ExecutionPolicy&& policy,
    const std::size_t depth,
    const BaseT& op1,
    const BaseT& op2,
    const std::size_t parallelFillThreshold)
{
    InnerContainerT pattern;
    
    // Compile-time branching for optimal memory allocation and initialization strategy
    if constexpr (std::default_initializable<BaseT> && std::copyable<BaseT>)
    {
        pattern.resize(depth * 2uz);

        auto beginIt = std::ranges::begin(pattern);
        auto midIt = std::ranges::next(beginIt, depth);
        auto endIt = std::ranges::end(pattern);

        // A configurable heuristic threshold for parallel array initialization.
        if ((depth * 2uz) >= parallelFillThreshold)
        {
            std::fill(std::forward<ExecutionPolicy>(policy), beginIt, midIt, op1);
            std::fill(std::forward<ExecutionPolicy>(policy), midIt, endIt, op2);
        }
        else
        {
            std::ranges::fill(beginIt, midIt, op1);
            std::ranges::fill(midIt, endIt, op2);
        }
    }
    else
    {
        // Safe fallback path: sequential emplace_back to avoid default construction
        pattern.reserve(depth * 2uz);

        for (std::size_t i = 0uz; i < depth; ++i)
        {
            pattern.emplace_back(op1);
        }
        for (std::size_t i = 0uz; i < depth; ++i)
        {
            pattern.emplace_back(op2);
        }
    }
    
    return pattern;
}

// ------------------------------------------------------------------
// Iteratively generate ALL balanced combinations.
// Uses if constexpr to dispatch between high-performance path 
// and the safe sequential fallback for complex objects.
// Introduces a configurable threshold to avoid parallelization overhead.
// ------------------------------------------------------------------
template <
    typename BaseT = DspOp,
    typename ContainerT = std::vector<std::vector<BaseT>>, 
    typename ExecutionPolicy>
    requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
ContainerT generateAllBalancedCombinations(
    ExecutionPolicy&& policy,
    const std::size_t depth,
    const BaseT& op1,
    const BaseT& op2,
    const std::size_t parallelFillThreshold = 16384uz)
{
    ContainerT combinations;
    
    // Calculate exact capacity to avoid dynamic reallocation during the loop
    const std::size_t exactCapacity = calculateBinomialCoefficient(depth * 2uz, depth);
    combinations.reserve(exactCapacity);

    // Automatically deduce the inner container type
    using InnerContainerT = typename ContainerT::value_type;
    
    InnerContainerT currentPattern = createInitialPattern<InnerContainerT, BaseT>(
        std::forward<ExecutionPolicy>(policy),
        depth,
        op1,
        op2,
        parallelFillThreshold
    );

    // Generate all permutations iteratively
    // Upgraded to C++20 std::ranges::next_permutation for perfect elegance
    do
    {
        combinations.emplace_back(currentPattern);
    } 
    while (std::ranges::next_permutation(currentPattern).found);

    return combinations;
}

// ------------------------------------------------------------------
// Generate a specific number of random balanced combinations.
// Used for extreme depths (e.g., depth=500) where exhaustive is impossible.
// Injects a customizable ShuffleOp functor to decouple the shuffling algorithm.
// ------------------------------------------------------------------
template <
    typename BaseT = DspOp,
    typename ContainerT = std::vector<std::vector<BaseT>>, 
    typename ExecutionPolicy, 
    typename Urbg,
    typename ShuffleOp = DefaultShuffleOp>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
        && std::uniform_random_bit_generator<std::remove_cvref_t<Urbg>>
        && std::invocable<
                std::remove_cvref_t<ShuffleOp>, 
                typename ContainerT::value_type&, 
                std::remove_cvref_t<Urbg>&
            >
ContainerT generateRandomBalancedCombinations(
    ExecutionPolicy&& policy,
    Urbg&& urbg, 
    const std::size_t depth, 
    const std::size_t numSamples,
    const BaseT& op1,
    const BaseT& op2,
    ShuffleOp&& shuffleOp = ShuffleOp{},
    const std::size_t parallelFillThreshold = 16384uz)
{
    using InnerContainerT = typename ContainerT::value_type;
    
    InnerContainerT basePattern = createInitialPattern<InnerContainerT, BaseT>(
        std::forward<ExecutionPolicy>(policy),
        depth,
        op1,
        op2,
        parallelFillThreshold
    );

    // 1. Pre-generate seeds sequentially because urbg state mutation is NOT thread-safe.
    using SeedT = std::invoke_result_t<Urbg&>;
    std::vector<SeedT> seeds;
    seeds.resize(numSamples);
    
    // Using C++23 std::ranges::generate instead of a raw loop
    std::ranges::generate(seeds, [&urbg]() { return urbg(); });

    // 2. Resize combinations array to allow independent parallel assignment
    ContainerT combinations;
    combinations.resize(numSamples);

    // 3. Map the seeds to fully randomized and shuffled patterns in strictly parallel manner
    // The shuffle algorithm is captured by value to ensure thread-safety across parallel executions.
    std::transform(
        std::forward<ExecutionPolicy>(policy),
        std::ranges::cbegin(seeds),
        std::ranges::cend(seeds),
        std::ranges::begin(combinations),
        [&basePattern, shuffleOp](const SeedT seed)
        {
            // Instantiate a local, thread-safe URBG using the unique pre-generated seed
            std::remove_cvref_t<Urbg> localRng(seed);
            
            InnerContainerT currentPattern = basePattern;
            
            // Invoke the injected generic shuffle operation
            shuffleOp(currentPattern, localRng);
            
            return currentPattern;
        }
    );

    return combinations;
}

// ------------------------------------------------------------------
// A callable struct (functor) to format a single test case into a string.
// This design inherently supports parallel execution architectures.
// ------------------------------------------------------------------
template <
    typename BaseT = DspOp, 
    typename ContainerT = std::vector<std::vector<BaseT>>>
struct TestCaseFormatter
{
    const ContainerT* opsList;
    std::size_t depth;
    bool isRandom;

    std::string operator()(const std::size_t index) const
    {
        // Use auto to seamlessly support any inner container type
        const auto& ops = (*opsList)[index];
        const std::string modeStr = isRandom ? "random_" : "exhaustive_";
        const std::string testName = 
            "test_" + modeStr + "depth_" + std::to_string(depth) + "_case_" + std::to_string(index);
        
        std::string os;
        
        // Pre-allocate buffer to avoid reallocation overhead during string appending
        os.reserve(1024uz);
        
        os += "// ------------------------------------------------------------------\n";
        os += "// Test Case: " + testName + " (Depth: " + std::to_string(depth) + ")\n";
        os += "// ------------------------------------------------------------------\n";
        os += "// CHECK-LABEL: func.func @" + testName + "\n";
        os += "// CHECK-SAME: (%[[ARG:.*]]: tensor<8x8xf32>)\n";
        os += "func.func @" + testName + "(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {\n";

        // Sequentially emit all MLIR operations
        // Use std::ranges::size to fully generalize the size resolution of the generic container
        const std::size_t opsCount = std::ranges::size(ops);
        for (std::size_t i = 0uz; i < opsCount; ++i)
        {
            const std::string_view opName = getOpName(ops[i]);
            const std::string inputName = (i == 0uz) ? "%arg0" : ("%" + std::to_string(i - 1uz));
            
            os += "  %";
            os += std::to_string(i);
            os += " = ";
            os += opName;
            os += " ";
            os += inputName;
            os += " : tensor<8x8xf32>\n";
        }

        os += "\n  // Ensure all operations are greedily eliminated\n";
        os += "  // CHECK-NOT: dsp.dct\n";
        os += "  // CHECK-NOT: dsp.idct\n\n";
        os += "  // The output should be directly wired to the input argument\n";
        os += "  // CHECK: return %[[ARG]] : tensor<8x8xf32>\n";
        os += "  return %";
        os += std::to_string(opsCount - 1uz);
        os += " : tensor<8x8xf32>\n";
        os += "}\n\n";

        return os;
    }
};

// ------------------------------------------------------------------
// Generate all cases based on threshold, process strings in parallel, 
// and sequentially write to the file stream.
// Constrained to accept standard execution policies using C++ concepts.
// ------------------------------------------------------------------
template <
    typename BaseT = DspOp,
    typename ContainerT = std::vector<std::vector<BaseT>>,
    typename ExecutionPolicy, 
    typename Urbg,
    typename ShuffleOp = DefaultShuffleOp>
    requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
          && std::uniform_random_bit_generator<std::remove_cvref_t<Urbg>>
          && std::invocable<
                 std::remove_cvref_t<ShuffleOp>, 
                 typename ContainerT::value_type&, 
                 std::remove_cvref_t<Urbg>&
             >
void generateAndWriteTests(
    ExecutionPolicy&& policy, 
    Urbg&& urbg,
    std::ostream& outFile, 
    const std::size_t depth,
    const BaseT& op1,
    const BaseT& op2,
    ShuffleOp&& shuffleOp = ShuffleOp{},
    const std::size_t parallelFillThreshold = 16384uz)
{
    constexpr std::size_t exhaustiveThreshold = 6uz;
    constexpr std::size_t randomSamplesCount = 1000uz;

    ContainerT allOps;
    bool isRandom = false;

    if (depth <= exhaustiveThreshold)
    {
        // Forward the execution policy and configurations
        allOps = generateAllBalancedCombinations<BaseT, ContainerT>(
            std::forward<ExecutionPolicy>(policy), 
            depth, 
            op1, 
            op2,
            parallelFillThreshold
        );
    }
    else
    {
        // Forward the execution policy, random engine, custom shuffle op and configurations
        allOps = generateRandomBalancedCombinations<BaseT, ContainerT>(
            std::forward<ExecutionPolicy>(policy), 
            std::forward<Urbg>(urbg), 
            depth, 
            randomSamplesCount, 
            op1, 
            op2,
            std::forward<ShuffleOp>(shuffleOp),
            parallelFillThreshold
        );
        
        isRandom = true;
    }
    
    // Resolve the total number of operations generically
    const std::size_t totalOpsCount = std::ranges::size(allOps);
    
    std::vector<std::string> testStrings(totalOpsCount);

    // Use C++20/C++23 std::views::iota to create a zero-cost lazy sequence of indices.
    // Notice the C++23 'uz' literal suffix for std::size_t!
    auto indicesView = std::views::iota(0uz, totalOpsCount);

    // Execute the string formatting strictly in parallel.
    std::transform(
        std::forward<ExecutionPolicy>(policy), 
        std::ranges::cbegin(indicesView), 
        std::ranges::cend(indicesView), 
        std::ranges::begin(testStrings), 
        TestCaseFormatter<BaseT, ContainerT>{&allOps, depth, isRandom}
    );

    // Sequentially flush parallel-generated strings to ensure file ordering
    for (const std::string& testStr : testStrings)
    {
        outFile << testStr;
    }
    
    const std::string modeStr = isRandom ? "Random sampled" : "Exhaustive";
    std::cout << "Depth " << depth << ": Auto-generated " << totalOpsCount 
              << " combinations (" << modeStr << ").\n";
}

// ------------------------------------------------------------------
// Main entry point for the parallel test generator
// ------------------------------------------------------------------
int main(const int argc, const char* const argv[])
{
    // Retrieve the output directory from command line arguments.
    // Fall back to a default relative path if not provided.
    std::string outputDir{"test/DSP"};
    if (argc > 1)
    {
        outputDir = argv[1];
    }

    // Adding 'uz' suffix indicates these are std::size_t constants
    constexpr std::array<std::size_t, 9> targetDepths = {
        2uz, 3uz, 4uz, 5uz, 6uz, 13uz, 50uz, 100uz, 500uz
    };

    // Instantiate the URBG exactly once in main to preserve its entropy state 
    // across multiple loop iterations and ensure maximum performance.
    std::mt19937_64 rng{std::random_device{}()};

    for (const std::size_t depth : targetDepths)
    {
        // Generate a separate file for each depth to maximize lit testing parallelism.
        // Output dynamically to the directory provided by the build system.
        const std::string outputFilename = 
            outputDir + "/canonicalize_exhaustive_depth_" + std::to_string(depth) + ".mlir";
            
        std::ofstream outFile(outputFilename);

        if (!outFile.is_open())
        {
            std::cerr << "Failed to open output file: " << outputFilename << "\n";
            return EXIT_FAILURE;
        }

        // Emit standard MLIR test directives
        outFile << "// RUN: dsp-opt %s --canonicalize | FileCheck %s\n\n";

        // Pass the operations dynamically from the highest level (main) down to the bottom.
        // We rely on the default values of BaseT, ContainerT, ShuffleOp, 
        // and parallelFillThreshold here to keep the call-site clean.
        generateAndWriteTests(
            std::execution::par_unseq, 
            rng, 
            outFile, 
            depth, 
            DspOp::DCT, 
            DspOp::IDCT
        );
    }

    std::cout << "Successfully generated all scalable tests.\n";

    return EXIT_SUCCESS;
}