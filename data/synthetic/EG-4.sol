contract ProcurementLottery {
    address[] public bidders;
    function bid() external { bidders.push(msg.sender); }
    function selectWinner() external view returns (address) {
        // SWC-120: predictable by the proposer
        uint256 r = uint256(blockhash(block.number - 1));
        return bidders[r % bidders.length];
    }
}