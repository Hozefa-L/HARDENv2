contract CitizenRegistry {
    address[] public citizens;
    mapping(address => uint8) public age;
    function register(address c, uint8 a) external {
        citizens.push(c); age[c] = a;
    }
    function countEligible(uint8 minAge)
        external view returns (uint256 n) {
        // SWC-128: scales with citizens.length, no bound
        for (uint i = 0; i < citizens.length; i++) {
            if (age[citizens[i]] >= minAge) n++;
        }
    }
}