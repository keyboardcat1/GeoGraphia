package com.github.keyboardcat1.geographia;

import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Holder;
import net.minecraft.server.level.WorldGenRegion;
import net.minecraft.world.level.*;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeManager;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.ChunkGenerator;
import net.minecraft.world.level.levelgen.*;
import net.minecraft.world.level.levelgen.blending.Blender;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

public class GeoGraphiaChunkGenerator extends ChunkGenerator {

    private static final Codec<Settings> SETTINGS_CODEC = RecordCodecBuilder.create(settingsInstance ->
            settingsInstance.group(
                    Codec.INT.fieldOf("base_height").forGetter(Settings::baseHeight),
                    Codec.FLOAT.fieldOf("vertical_scale").forGetter(Settings::verticalVariance),
                    Codec.FLOAT.fieldOf("horizontal_scale").forGetter(Settings::horizontalVariance)
            ).apply(settingsInstance, Settings::new)
    );
    public static final Codec<GeoGraphiaChunkGenerator> CODEC = RecordCodecBuilder.create(geoGraphiaChunkGeneratorInstance ->
        geoGraphiaChunkGeneratorInstance.group(
            BiomeSource.CODEC.fieldOf("biome_source").forGetter(GeoGraphiaChunkGenerator::getBiomeSource),
            SETTINGS_CODEC.fieldOf("settings").forGetter(GeoGraphiaChunkGenerator::getSettings)
        ).apply(geoGraphiaChunkGeneratorInstance, GeoGraphiaChunkGenerator::new)
    );

    private final Settings settings;

    public GeoGraphiaChunkGenerator(BiomeSource biomeSource, Settings settings) {
        super(biomeSource);
        this.settings = settings;
    }

    public Settings getSettings() {
        return settings;
    }

    @Override
    protected @NotNull Codec<? extends ChunkGenerator> codec() {
        return CODEC;
    }

    @Override
    public void buildSurface(@NotNull WorldGenRegion level, @NotNull StructureManager structureManager, @NotNull RandomState randomState, ChunkAccess chunk) {
        BlockState bedrock = Blocks.BEDROCK.defaultBlockState();
        BlockState stone = Blocks.STONE.defaultBlockState();
        BlockState water = Blocks.WATER.defaultBlockState();
        ChunkPos chunkpos = chunk.getPos();

        BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
        int x;
        int z;

        for (x = 0; x < 16; x++) for (z = 0; z < 16; z++) {
            int realX = chunkpos.x * 16 + x;
            int realZ = chunkpos.z * 16 + z;
            chunk.setBlockState(pos.set(x, getMinY(), z), bedrock, false);

            int height = getHeightAt(realX, realZ);
            for (int y = getMinY()+1 ; y < height ; y++) {
                if (height <= getSeaLevel()) {
                    chunk.setBlockState(pos.set(x, y, z), water, false);
                } else {
                    chunk.setBlockState(pos.set(x, y, z), stone, false);
                }
            }
        }
    }
    private int getHeightAt(int x, int z) {
        final float w = GlobalHeights.WIDTH;
        final float h = GlobalHeights.HEIGHT;
        if (x < -Math.floor(w/2F) || x >= Math.ceil(w/2F) || z < -Math.floor(h/2F) || z >= Math.ceil(h/2F)) return getSeaLevel();
        int arrX = x + (int)Math.ceil((float)GlobalHeights.WIDTH/2F);
        int arrZ = z + (int)Math.ceil((float)GlobalHeights.HEIGHT/2F);
        return getSeaLevel() + (int)(settings.verticalVariance * GlobalHeights.getHeight(arrX, arrZ));
    }

    @Override
    public int getBaseHeight(int x, int z, Heightmap.@NotNull Types types, @NotNull LevelHeightAccessor levelHeightAccessor,
                             @NotNull RandomState randomState) {
        return getHeightAt(x, z);
    }
    @Override
    public @NotNull NoiseColumn getBaseColumn(int x, int z, @NotNull LevelHeightAccessor levelHeightAccessor, @NotNull RandomState randomState) {
        int y = getBaseHeight(x, z, Heightmap.Types.WORLD_SURFACE_WG, levelHeightAccessor, randomState) - getMinY();
        if (y==0)
            return new NoiseColumn(getMinY(), new BlockState[]{});
        BlockState stone = Blocks.STONE.defaultBlockState();
        BlockState[] states = new BlockState[y];
        Arrays.fill(states, stone);
        states[0] = Blocks.BEDROCK.defaultBlockState();
        return new NoiseColumn(getMinY(), states);
    }

    @Override
    public void applyCarvers(@NotNull WorldGenRegion level, long seed, @NotNull RandomState randomState, @NotNull BiomeManager biomeManager,
                             @NotNull StructureManager structureManager, @NotNull ChunkAccess chunk, GenerationStep.@NotNull Carving carving) {
    }

    @Override
    public void spawnOriginalMobs(WorldGenRegion level) {
        ChunkPos chunkpos = level.getCenter();
        Holder<Biome> biomeHolder = level.getBiome(chunkpos.getWorldPosition().atY(level.getMaxBuildHeight() - 1));
        WorldgenRandom worldgenrandom = new WorldgenRandom(new LegacyRandomSource(RandomSupport.generateUniqueSeed()));
        worldgenrandom.setDecorationSeed(level.getSeed(), chunkpos.getMinBlockX(), chunkpos.getMinBlockZ());
        NaturalSpawner.spawnMobsForChunkGeneration(level, biomeHolder, chunkpos, worldgenrandom);
    }

    @Override
    public @NotNull CompletableFuture<ChunkAccess> fillFromNoise(@NotNull Executor executor, @NotNull Blender blender, @NotNull RandomState randomState,
                                                                 @NotNull StructureManager structureManager, @NotNull ChunkAccess chunkAccess) {
        return CompletableFuture.completedFuture(chunkAccess);
    }

    @Override
    public int getGenDepth() {
        return 320;
    }
    @Override
    public int getSeaLevel() {
        return settings.baseHeight;
    }
    @Override
    public int getMinY() {
        return -64;
    }

    @Override
    public void addDebugScreenInfo(@NotNull List<String> strings, @NotNull RandomState randomState, @NotNull BlockPos blockPos) {
    }

    public record Settings(int baseHeight, float verticalVariance, float horizontalVariance) {}
}
